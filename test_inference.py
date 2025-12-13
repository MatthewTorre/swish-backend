from model import load_model, predict_video

MODEL_PATH = "baseline_model.pth"   # adjust path if needed
TEST_VIDEO = "test_clip.mp4"        # any short basketball clip

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    result = predict_video(model, TEST_VIDEO)
    print(result)
