require 'opencv'
require 'deepface'

include OpenCV

face_cascade = CvHaarClassifierCascade::load('path_to_haarcascade_frontalface_default.xml')
emotion_model = DeepFace::FacialExpression.new

capture = CvCapture.open
window = GUI::Window.new('Facial Emotion Recognition')

while true
  frame = capture.query
  gray_frame = frame.BGR2GRAY
  faces = gray_frame.detect_objects(face_cascade)

  faces.each do |face|
    face_region = frame.sub_rect(face)
    emotions = emotion_model.predict_emotion(face_region)
    text_position = CvPoint.new(face.x, face.y - 10)
    frame.put_text!(emotions['dominant_emotion'], text_position, CV_FONT_HERSHEY_SIMPLEX, 1, CvColor::Red)
  end

  window.show(frame)

  break if GUI::wait_key(10) == 27
end

capture.release
window.destroy
