#include "mediapipe_api.h"
#include "mediapipe_interface.hpp"
#include "mediapipe/framework/port/opencv_core_inc.h"

FaceMeshInterface* face_mesh_interface = nullptr;
HandTrackInterface* hand_track_interface = nullptr;
PoseTrackInterface* pose_track_interface = nullptr;
HolisticTrackInterface* holistic_track_interface = nullptr;
FaceBlendShapeInterface* face_blend_shape_interface = nullptr;

LibraryExport void CreateFaceMeshInterface(const char * graph_name) {
    face_mesh_interface = new FaceMeshInterface();
    face_mesh_interface->SetGraph(graph_name);
}

LibraryExport void ReleaseFaceMeshInterface() {
    delete face_mesh_interface;
    face_mesh_interface = nullptr;
}

LibraryExport void StartFaceMesh() {
    face_mesh_interface->Start();
}

LibraryExport void FaceMeshProcess(void * mat) {
    auto cpp_mat = static_cast<cv::Mat*>(mat);
    face_mesh_interface->Process(*cpp_mat);
}

LibraryExport void SetFaceMeshObserveCallback(landmark_callback callback) {
    face_mesh_interface->SetObserveCallback(callback);
}

LibraryExport void ObserveFaceMesh() {
    face_mesh_interface->Observe();
}

LibraryExport void AddFaceMeshPoller() {
    face_mesh_interface->AddOutputStreamPoller();
}

LibraryExport void GetFaceMeshOutput(NormalizedLandmark * normalized_landmark_list, unsigned size) {
    face_mesh_interface->GetOutput(normalized_landmark_list, size);
}

LibraryExport void StopFaceMesh() {
    face_mesh_interface->Stop();
}

LibraryExport void CreateHandTrackInterface(const char * graph_name) {
    hand_track_interface = new HandTrackInterface();
    hand_track_interface->SetGraph(graph_name);
}

LibraryExport void ReleaseHandTrackInterface() {
    delete hand_track_interface;
    hand_track_interface = nullptr;
}

LibraryExport void StartHandTrack() {
    hand_track_interface->Start();
}

LibraryExport void HandTrackProcess(void * mat) {
    auto cpp_mat = static_cast<cv::Mat*>(mat);
    hand_track_interface->Process(*cpp_mat);
}

LibraryExport void SetHandTrackObserveCallback(landmark_callback callback) {
    hand_track_interface->SetObserveCallback(callback);
}

LibraryExport void ObserveHandTrack() {
    hand_track_interface->Observe();
}

LibraryExport void AddHandTrackPoller() {
    hand_track_interface->AddOutputStreamPoller();
}

LibraryExport void GetHandTrackOutput(NormalizedLandmark * normalized_landmark_list, unsigned size) {
    hand_track_interface->GetOutput(normalized_landmark_list, size);
}

LibraryExport void StopHandTrack() {
    hand_track_interface->Stop();
}

LibraryExport void CreatePoseTrackInterface(const char * graph_name) {
    pose_track_interface = new PoseTrackInterface();
    pose_track_interface->SetGraph(graph_name);
}

LibraryExport void ReleasePoseTrackInterface() {
    delete pose_track_interface;
    pose_track_interface = nullptr;
}

LibraryExport void StartPoseTrack() {
    pose_track_interface->Start();
}

LibraryExport void PoseTrackProcess(void * mat) {
    auto cpp_mat = static_cast<cv::Mat*>(mat);
    pose_track_interface->Process(*cpp_mat);
}

LibraryExport void SetPoseTrackObserveCallback(landmark_callback callback) {
    pose_track_interface->SetObserveCallback(callback);
}

LibraryExport void ObservePoseTrack() {
    pose_track_interface->Observe();
}

LibraryExport void AddPoseTrackPoller() {
    pose_track_interface->AddOutputStreamPoller();
}

LibraryExport void GetPoseTrackOutput(NormalizedLandmark * normalized_landmark_list, unsigned size) {
    pose_track_interface->GetOutput(normalized_landmark_list, size);
}

LibraryExport void StopPoseTrack() {
    pose_track_interface->Stop();
}

LibraryExport void CreateHolisticTrackInterface(const char * graph_name) {
    holistic_track_interface = new HolisticTrackInterface();
    holistic_track_interface->SetGraph(graph_name);
}

LibraryExport void ReleaseHolisticTrackInterface() {
    delete holistic_track_interface;
    holistic_track_interface = nullptr;
}

LibraryExport void StartHolisticTrack() {
    holistic_track_interface->Start();
}

LibraryExport void HolisticTrackProcess(void * mat) {
    auto cpp_mat = static_cast<cv::Mat*>(mat);
    holistic_track_interface->Process(*cpp_mat);
}

LibraryExport void SetHolisticTrackObserveCallback(landmark_callback callback, HolisticCallbackType type) {
    holistic_track_interface->SetObserveCallback(callback, type);
}

LibraryExport void ObserveHolisticTrack() {
    holistic_track_interface->Observe();
}

LibraryExport void StopHolisticTrack() {
    holistic_track_interface->Stop();
}

LibraryExport void CreateFaceBlendShapeInterface(const char * graph_name) {
    face_blend_shape_interface  = new FaceBlendShapeInterface();
    face_blend_shape_interface->SetGraph(graph_name);
}

LibraryExport void ReleaseFaceBlendShapeInterface(){
    delete face_blend_shape_interface;
    face_blend_shape_interface = nullptr;
}

LibraryExport void StartFaceBlendShape() {
    face_blend_shape_interface->Start();
}

LibraryExport void FaceBlendShapeProcess(void * mat) {
    auto cpp_mat = static_cast<cv::Mat*>(mat);
    face_blend_shape_interface->Process(*cpp_mat);
}

LibraryExport void SetFaceBlendShapeCallback(blend_shape_callback callback) {
    face_blend_shape_interface->SetObserveCallback(callback);
}

LibraryExport void ObserveFaceBlendShape() {
    face_blend_shape_interface->Observe();
}

LibraryExport void AddFaceBlendShapePoller() {
    face_blend_shape_interface->AddOutputStreamPoller();
}

LibraryExport void GetFaceBlendShapeOutput(float * blend_shape_list, unsigned size) {
    face_blend_shape_interface->GetOutput(blend_shape_list, size);
}

LibraryExport void StopFaceBlendShape() {
    face_blend_shape_interface->Stop();
}
