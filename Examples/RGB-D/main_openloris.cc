/**
 * NGD-SLAM evaluation entry point for the OpenLORIS-Scene dataset.
 *
 * Invocation by run_openloris.sh:
 *   main_openloris <config_yaml> <kernel> <dynamic_filtering>
 *                  <pangolin_viewer> <object_detection_viewer> <package_path>
 *
 * The per-run <config_yaml> contains:
 *   settings_file:          path to camera/ORB settings YAML (e.g. input_data/openloris.yaml)
 *   sequence_path:          path to the sequence directory (contains rgb/ and depth/ sub-dirs)
 *   association_file:       path to association.txt
 *   robust_kernel:          kernel name (Huber | Barron | BarronSignalHuber) — logged only
 *   dynamic_filtering:      0 or 1  — logged only
 *   object_detection_viewer: 0 or 1 — reserved
 *
 * Outputs (written to CWD, i.e. package_path):
 *   CameraTrajectory.txt    TUM-format full trajectory
 *   KeyFrameTrajectory.txt  TUM-format key-frame trajectory
 *   alpha_c_log.txt         provenance log (kernel, filtering flag)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <unistd.h>

#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;

static void LoadImages(const string &assocFile,
                       vector<string> &vRGB,
                       vector<string> &vD,
                       vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if (argc < 2) {
        cerr << "\nUsage: main_openloris <config_yaml> [kernel] [dynamic_filtering]"
                " [pangolin_viewer] [object_detection_viewer] [package_path]\n\n";
        return 1;
    }

    // ── 1. Read per-run YAML config ────────────────────────────────────────
    cv::FileStorage fsCfg(argv[1], cv::FileStorage::READ);
    if (!fsCfg.isOpened()) {
        cerr << "Error: cannot open config file: " << argv[1] << "\n";
        return 1;
    }
    string settingsFile    = (string)fsCfg["settings_file"];
    string sequencePath    = (string)fsCfg["sequence_path"];
    string associationFile = (string)fsCfg["association_file"];
    string robustKernel    = (string)fsCfg["robust_kernel"];
    int    dynamicFilter   = (int)   fsCfg["dynamic_filtering"];
    int    objDetViewer    = (int)   fsCfg["object_detection_viewer"];
    fsCfg.release();

    // ── 2. Apply CLI overrides (run_openloris.sh passes these) ─────────────
    if (argc >= 3) robustKernel  = argv[2];
    if (argc >= 4) dynamicFilter = stoi(argv[3]);
    bool bViewer = true;
    if (argc >= 5) bViewer = (stoi(argv[4]) != 0);
    if (argc >= 6) objDetViewer  = stoi(argv[5]);

    // chdir to package root BEFORE constructing System so that YOLO relative
    // paths (./Thirdparty/YOLO/…) and Vocabulary/ORBvoc.txt resolve correctly.
    if (argc >= 7) {
        if (chdir(argv[6]) != 0)
            cerr << "Warning: chdir to package_path failed: " << argv[6] << "\n";
    }

    cout << "\n=== OpenLORIS-Scene evaluation ===\n"
         << "  settings_file:           " << settingsFile    << "\n"
         << "  sequence_path:           " << sequencePath    << "\n"
         << "  association_file:        " << associationFile << "\n"
         << "  robust_kernel:           " << robustKernel    << "\n"
         << "  dynamic_filtering:       " << dynamicFilter   << "\n"
         << "  pangolin_viewer:         " << bViewer         << "\n"
         << "  object_detection_viewer: " << objDetViewer    << "\n\n";

    // ── 3. Load RGB-D association list ─────────────────────────────────────
    vector<string> vRGB, vD;
    vector<double> vTimestamps;
    LoadImages(associationFile, vRGB, vD, vTimestamps);

    int nImages = (int)vRGB.size();
    if (nImages == 0) {
        cerr << "Error: no images found in association file: " << associationFile << "\n";
        return 1;
    }
    if ((int)vD.size() != nImages) {
        cerr << "Error: mismatch between RGB and depth image counts.\n";
        return 1;
    }

    // ── 4. Initialise NGD-SLAM ─────────────────────────────────────────────
    // Vocabulary lives at <package_path>/Vocabulary/ORBvoc.txt.
    // After chdir above, this relative path resolves correctly.
    const string vocabFile = "Vocabulary/ORBvoc.txt";

    ORB_SLAM3::System SLAM(vocabFile, settingsFile,
                           ORB_SLAM3::System::RGBD, bViewer);
    float imageScale = SLAM.GetImageScale();

    // ── 5. Main tracking loop ──────────────────────────────────────────────
    vector<float> vTimesTrack(nImages);
    cout << "Start processing sequence (" << nImages << " frames)…\n\n";

    cv::Mat imRGB, imD;
    for (int ni = 0; ni < nImages; ++ni) {
        imRGB = cv::imread(sequencePath + "/" + vRGB[ni], cv::IMREAD_UNCHANGED);
        imD   = cv::imread(sequencePath + "/" + vD[ni],   cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imRGB.empty()) {
            cerr << "Error: failed to load RGB: "
                 << sequencePath << "/" << vRGB[ni] << "\n";
            return 1;
        }
        if (imD.empty()) {
            cerr << "Error: failed to load depth: "
                 << sequencePath << "/" << vD[ni] << "\n";
            return 1;
        }

        if (imageScale != 1.f) {
            int w = static_cast<int>(imRGB.cols * imageScale);
            int h = static_cast<int>(imRGB.rows * imageScale);
            cv::resize(imRGB, imRGB, cv::Size(w, h));
            cv::resize(imD,   imD,   cv::Size(w, h));
        }

#ifdef COMPILEDWITHC11
        auto t1 = std::chrono::steady_clock::now();
#else
        auto t1 = std::chrono::monotonic_clock::now();
#endif
        SLAM.TrackRGBD(imRGB, imD, tframe);
#ifdef COMPILEDWITHC11
        auto t2 = std::chrono::steady_clock::now();
#else
        auto t2 = std::chrono::monotonic_clock::now();
#endif
        double ttrack = std::chrono::duration_cast<
            std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = static_cast<float>(ttrack);

        // Real-time pacing: wait if tracking finished faster than frame period.
        double T = 0.0;
        if (ni < nImages - 1)  T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)       T = tframe - vTimestamps[ni - 1];
        if (ttrack < T) usleep(static_cast<useconds_t>((T - ttrack) * 1e6));
    }

    SLAM.Shutdown();

    // ── 6. Timing statistics ───────────────────────────────────────────────
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ++ni) totaltime += vTimesTrack[ni];
    cout << "-------\n"
         << "Median tracking time: " << vTimesTrack[nImages / 2] << " s\n"
         << "Mean   tracking time: " << totaltime / nImages       << " s\n";

    // ── 7. Save trajectories and provenance log ────────────────────────────
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    {
        ofstream fLog("alpha_c_log.txt");
        fLog << "robust_kernel: "     << robustKernel << "\n"
             << "dynamic_filtering: " << dynamicFilter << "\n"
             << "sequence_path: "     << sequencePath  << "\n";
    }

    return 0;
}

// -----------------------------------------------------------------------------
// LoadImages: parse a TUM-style association file
//   format: <rgb_ts> <rgb_path> <depth_ts> <depth_path>
// -----------------------------------------------------------------------------
static void LoadImages(const string &assocFile,
                       vector<string> &vRGB,
                       vector<string> &vD,
                       vector<double> &vTimestamps)
{
    ifstream f(assocFile.c_str());
    if (!f.is_open()) {
        cerr << "Error: cannot open association file: " << assocFile << "\n";
        return;
    }
    while (!f.eof()) {
        string s;
        getline(f, s);
        if (s.empty() || s[0] == '#') continue;
        istringstream ss(s);
        double t1, t2;
        string sRGB, sD;
        if (!(ss >> t1 >> sRGB >> t2 >> sD)) continue;
        vTimestamps.push_back(t1);
        vRGB.push_back(sRGB);
        vD.push_back(sD);
    }
}
