import streamlit as st
import subprocess
import cv2
from PIL import Image
import os

def main():
    st.title("Digitalization of Entry/Exit System for IIT Guwahati")

    st.sidebar.title("Categories")
    category = st.sidebar.radio("Choose a category", ("Original Method", "Demo"))

    if category == "Original Method":
        original_method()

    elif category == "Demo":
        demo()

def original_method():
    st.header("Original Method")

    run_live_recording()

    run_image_capture()

    display_output()

def demo():
    st.header("Demo")

    uploaded_video = run_video_upload()

    if uploaded_video is not None:
        st.write("Video uploaded successfully.")
        st.video(uploaded_video)
        
        # Execute command
        execute_command(uploaded_video)

    uploaded_image = run_image_upload()
    if uploaded_image is not None:
        st.write("Image uploaded successfully.")
        st.image(uploaded_image, use_column_width=True)
        save_uploaded_image(uploaded_image)
        run_custom_command()

    display_output()

def run_live_recording():
    st.subheader("Live Video Recording")
    run = st.checkbox("Start Recording")

    if run:
        st.write("Recording started...")
        command = r'python detect.py --weights runs/train/exp3/weights/last.pt --source 0'

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        # cap = cv2.VideoCapture(0)
        # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (640, 480))

        # while run:
        #     ret, frame = cap.read()
        #     if ret:
        #         out.write(frame)
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         st.image(Image.fromarray(frame), use_column_width=True)
                
        #     else:
        #         st.write("Error capturing frame.")
        #         break

        #     if not run:
        #         break

        # st.write("Recording stopped.")
        # out.release()
        # cap.release()

def run_image_capture():
    st.subheader("Image Capture")
    capture = st.button("Capture Image")

    if capture:
        st.write("Capturing image...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(frame), use_column_width=True)
            st.write("Image captured.")
            #save_captured_image(frame)
        else:
            st.write("Error capturing image.")
        cap.release()

def run_video_upload():
    st.subheader("Video Upload")
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    return uploaded_video

def run_image_upload():
    st.subheader("Image Upload")
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
    return uploaded_image

def display_output():
    st.subheader("Output")
    st.write("Processed results will be displayed here.")

def execute_command(uploaded_video):
    # Specify the command
    command = f"python detect_my.py --weights runs/train/exp4/weights/best.pt --img 640 --conf 0.25 --source E:\\IPML\\input_output\\{uploaded_video.name}"
    
    # Execute the command using subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    
    # Print the output and error
    if output:
        st.write("Command output:")
        st.code(output.decode("utf-8"))
    if error:
        st.write("Command error:")
        st.code(error.decode("utf-8"))

def save_uploaded_image(uploaded_image):
    # Save the uploaded image to a specified directory
    if not os.path.exists("uploaded_images"):
        os.makedirs("uploaded_images")
    
    image_path = os.path.join("uploaded_images", "uploaded_image.png")
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    st.write("Image saved successfully.")

def save_captured_image(captured_frame):
    # Save the captured image to a specified directory with the name "live.png"
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")
    
    image_path = os.path.join("captured_images", "live.png")
    cv2.imwrite(image_path, cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR))
    st.write("Image saved successfully.")

def run_custom_command():
    # Run the custom command
    # command = r'e:/IPML/venv/Scripts/python.exe e:/IPML/my_part.py'
    command = r'python my_part.py'

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # Print the output and error
    if output:
        st.write("Custom command output:")
        st.code(output.decode("utf-8"))
    if error:
        st.write("Custom command error:")
        st.code(error.decode("utf-8"))

if __name__ == "__main__":
    main()
