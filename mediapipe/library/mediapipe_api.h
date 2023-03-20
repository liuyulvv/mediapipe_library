#ifndef MEDIAPIPE_API_H_
#define MEDIAPIPE_API_H_

#ifdef _WIN32
#ifndef LibraryExport
#define LibraryExport __declspec(dllexport)
#endif
#else
#ifndef LibraryExport
#define LibraryExport __attribute__((visibility("default")))
#endif
#endif

#include "mediapipe_interface.h"

LibraryExport FaceDetectInterface* CreateFaceDetectInterface();

LibraryExport FaceMeshInterface* CreateFaceMeshInterface();

LibraryExport HandTrackInterface* CreateHandTrackInterface();

LibraryExport PoseTrackInterface* CreatePoseTrackInterface();

LibraryExport HolisticTrackInterface* CreateHolisticTrackInterface();

#endif