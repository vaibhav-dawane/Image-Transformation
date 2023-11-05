import cv2
import numpy as np
import streamlit as st

# Create a Streamlit app
st.title('Image Transformations')

# Add an image upload option
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    st.write("Please upload an image.")

# Transformation options in the sidebar
transformation = st.sidebar.selectbox("Select Transformation", ["Original", "Translation", "Rotation", "Scaling", "Shearing", "Perspective", "Elastic Distortion"])

# Apply and display the selected transformation with user input
if uploaded_image is not None:
    if transformation == "Original":
        st.image(image, caption='Original Image', use_column_width=True, channels="BGR")
    else:
        if transformation == "Translation":
            translation_x = st.number_input("X Translation (pixels)", -500, 500, 100)
            translation_y = st.number_input("Y Translation (pixels)", -500, 500, 50)
            translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            transformed_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        elif transformation == "Rotation":
            rotation_angle = st.number_input("Rotation Angle (degrees)", -180, 180, 30)
            rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), rotation_angle, 1)
            transformed_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        elif transformation == "Scaling":
            scaling_x = st.number_input("X Scaling Factor", 0.1, 2.0, 1.2)
            scaling_y = st.number_input("Y Scaling Factor", 0.1, 2.0, 0.8)
            scaling_matrix = np.float32([[scaling_x, 0, 0], [0, scaling_y, 0]])
            transformed_image = cv2.warpAffine(image, scaling_matrix, (image.shape[1], image.shape[0]))
        elif transformation == "Shearing":
            shearing_x = st.number_input("X Shearing Factor", -1.0, 1.0, 0.2)
            shearing_y = st.number_input("Y Shearing Factor", -1.0, 1.0, 0.2)
            shearing_matrix = np.float32([[1, shearing_x, 0], [shearing_y, 1, 0]])
            transformed_image = cv2.warpAffine(image, shearing_matrix, (image.shape[1], image.shape[0]))
        elif transformation == "Perspective":
            st.write("Specify the coordinates of four corners for perspective transformation:")
            col1, col2 = st.columns(2)  # Split the layout into two columns

            with col1:
                top_left_x = st.number_input("Top-left X", 0.0, float(image.shape[1] - 1), 50.0, 1.0)  # Ensure that min_value, max_value, and step are all float
                top_left_y = st.number_input("Top-left Y", 0.0, float(image.shape[0] - 1), 50.0, 1.0)  # Ensure that min_value, max_value, and step are all float
                bottom_left_x = st.number_input("Bottom-left X", 0.0, float(image.shape[1] - 1), 50.0, 1.0)  # Ensure that min_value, max_value, and step are all float
                bottom_left_y = st.number_input("Bottom-left Y", 0.0, float(image.shape[0] - 1), 200.0, 1.0)  # Ensure that min_value, max_value, and step are all float

            with col2:
                top_right_x = st.number_input("Top-right X", 0.0, float(image.shape[1] - 1), 200.0, 1.0)  # Ensure that min_value, max_value, and step are all float
                top_right_y = st.number_input("Top-right Y", 0.0, float(image.shape[0] - 1), 50.0, 1.0)  # Ensure that min_value, max_value, and step are all float
                bottom_right_x = st.number_input("Bottom-right X", 0.0, float(image.shape[1] - 1), 200.0, 1.0)  # Ensure that min_value, max_value, and step are all float
                bottom_right_y = st.number_input("Bottom-right Y", 0.0, float(image.shape[0] - 1), 200.0, 1.0)  # Ensure that min_value, max_value, and step are all float

            # Form pairs of coordinates for pts1
            pts1 = np.float32([
                [top_left_x, top_left_y],
                [top_right_x, top_right_y],
                [bottom_left_x, bottom_left_y],
                [bottom_right_x, bottom_right_y]
            ])
            
            pts2 = np.float32([[10, 10], [image.shape[1] - 10, 10], [10, image.shape[0] - 10], [image.shape[1] - 10, image.shape[0] - 10]])  # Destination points
            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            transformed_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))


        elif transformation == "Elastic Distortion":
            rows, cols, _ = image.shape
            grid_size = 20
            x, y = np.meshgrid(np.linspace(0, cols, grid_size), np.linspace(0, rows, grid_size))
            x_displace = st.slider("X Displacement (pixels)", -10, 10, 0)
            y_displace = st.slider("Y Displacement (pixels)", -10, 10, 0)
            x_displace = x_displace * np.sin(2 * np.pi * x / cols)
            y_displace = y_displace * np.sin(2 * np.pi * y / rows)
            map_x = (x + x_displace).astype(np.float32)
            map_y = (y + y_displace).astype(np.float32)
            transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        st.image(transformed_image, caption=f'{transformation} Image', use_column_width=True, channels="BGR")
