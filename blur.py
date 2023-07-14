from PIL import Image, ImageFilter
def add_blurry_rotation(image_path, output_path, angle_degrees):

    # Open the image

    image = Image.open(image_path)

 

    # Convert the angle from degrees to radians

    angle_radians = angle_degrees * (3.14159 / 180.0)

 

    # Rotate the image

    rotated_image = image.rotate(angle_degrees, resample=Image.BILINEAR, expand=True, fillcolor='white')

 

    # Apply a blur filter

    blurred_image = rotated_image.filter(ImageFilter.GaussianBlur(radius=2))

 

    # Calculate the output image size

    output_size = rotated_image.size

 

    # Create a transparent background image

    transparent_image = Image.new("RGBA", output_size, (0, 0, 0, 0))

 

    # Paste the blurred image onto the transparent background

    transparent_image.paste(blurred_image, (0, 0), mask=blurred_image)

 

    # Save the resulting image

    transparent_image.save(output_path)

 

# Example usage

input_image_path = './datasets/model1_data/still/j_0480_001.png'

output_image_path = 'output.png'

rotation_angle = 15  # Angle in degrees

 

add_blurry_rotation(input_image_path, output_image_path, rotation_angle)

