#include "mediapipe_api.h"
#include "mediapipe_library.h"

LibraryExport FaceDetectInterface* CreateFaceDetectInterface() {
    FaceDetectInterface* interface = new FaceDetectLibrary();
    return interface;
}

LibraryExport FaceMeshInterface* CreateFaceMeshInterface() {
    FaceMeshInterface* interface  = new FaceMeshLibrary();
    return interface;
}

LibraryExport HandTrackInterface * CreateHandTrackInterface() {
    HandTrackInterface* interface = new HandTrackLibrary();
    return interface;
}

LibraryExport PoseTrackInterface * CreatePoseTrackInterface() {
    PoseTrackInterface* interface = new PoseTrackLibrary();
    return interface;
}

LibraryExport HolisticTrackInterface * CreateHolisticTrackInterface() {
    HolisticTrackInterface* interface = new HolisticTrackLibrary();
    return interface;
}
