import roslibpy
import time

import cv2
import base64

import numpy as np
from cv_bridge import CvBridge

bridge = CvBridge()

# just a test to see if I can subscribe to a topic from a conda env

# subribe to the topic /chatter for now


def test_roslibpy():
    # Create a Roslibpy client
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    # Create a subscriber to the /chatter topic
    subscriber = roslibpy.Topic(client, "/chatter", "std_msgs/String")

    # Define a callback function to handle incoming messages
    def callback(message):
        print("Received message: {}".format(message["data"]))

    print("Waiting for messages on /chatter...")
    # Subscribe to the topic with the callback function
    subscriber.subscribe(callback)

    # Keep the script running to listen for messages
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # Unsubscribe and close the client
    subscriber.unsubscribe()
    client.terminate()


def test_rosliby_publisher():

    # Create a Roslibpy client
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    # Create a publisher to the /chatter topic
    publisher = roslibpy.Topic(client, "/chatter", "std_msgs/String")

    try:
        while True:
            # Publish a message every 500ms
            message = roslibpy.Message({"data": "Hello from the publisher!"})
            publisher.publish(message)
            print("Published message: {}".format(message["data"]))
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    # Close the client
    client.terminate()


def test_roslibpy_subscriber_image():
    # Create a Roslibpy client
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    # Create a subscriber to the /camera/image topic
    subscriber = roslibpy.Topic(client, "/camera/image", "sensor_msgs/Image")

    # Define a callback function to handle incoming messages
    def callback(message):
        print("Received image message.")
        # Decode the image data from the ROS message
        base64_bytes = message["data"].encode("ascii")
        image_bytes = base64.b64decode(base64_bytes)
        # Convert the byte data to a NumPy array
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        # Reshape the NumPy array to the image dimensions
        width = message["width"]
        height = message["height"]
        channels = message["encoding"]
        if channels == "bgr8":
            cv_image = np_array.reshape((height, width, 3))
        elif channels == "rgb8":
            cv_image = np_array.reshape((height, width, 3))[:, :, ::-1]
        else:
            print(f"Unsupported image encoding: {channels}")
            return

        cv2.imshow("Received Image", cv_image)
        cv2.waitKey(1)

    print("Waiting for image messages on /camera/image...")
    # Subscribe to the topic with the callback function
    subscriber.subscribe(callback)

    # Keep the script running to listen for messages
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # Unsubscribe and close the client
    subscriber.unsubscribe()
    client.terminate()


def roslibpy_service_image():
    # Create a Roslibpy client
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    # Create a service server for the /process_image service
    #! service = roslibpy.Service(client, '/process_image', 'ibpc_interfaces/srv/GetPoseEstimates')
    service = roslibpy.Service(
        client, "/process_image", "ibpc_interfaces/srv/TestService"
    )

    # Define a callback function to handle service requests
    def handle_request(request, response):
        print("Received service request to process an image.")

        # Decode the base64 image from the request
        #! base64_image = request['image']
        image_message = request["rgb"]
        print("Received image message:", image_message)
        base64_bytes = image_message["data"].encode("ascii")
        image_bytes = base64.b64decode(base64_bytes)
        # Convert the byte data to a NumPy array
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        width = image_message["width"]
        height = image_message["height"]
        channels = image_message["encoding"]
        if channels == "bgr8":
            cv_image = np_array.reshape((height, width, 3))
        elif channels == "rgb8":
            cv_image = np_array.reshape((height, width, 3))[:, :, ::-1]
        else:
            print(f"Unsupported image encoding: {channels}")
            return
        # show the image
        cv2.imshow("Received Image", cv_image)
        cv2.waitKey(1)

        # do nothing with the image for now

        # Process the image (example: convert to grayscale)
        # processed_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Encode the processed image back to base64
        # _, buffer = cv2.imencode('.jpg', processed_image)
        # base64_processed_image = base64.b64encode(buffer).decode('ascii')

        # Populate the response
        # response['success'] = True
        # response['processed_image'] = base64_processed_image
        # response['message'] = 'Image processed successfully.'

        print("Processed image and sent response.")
        # Ensure the response includes a 'success' field with a boolean value
        # response['success'] = True
        # Convert the response to a dictionary before returning

        # ? that is how we set the response !!!!!
        response["success"] = True
        return True

    # Advertise the service with the callback function
    service.advertise(handle_request)

    print("Service /process_image is now available.")

    # Keep the script running to handle service requests
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # Unadvertise the service and close the client
    service.unadvertise()
    client.terminate()


if __name__ == "__main__":

    # both sides work !
    # test_roslibpy()
    # test_rosliby_publisher()

    # image tests
    # test_roslibpy_subscriber_image()
    roslibpy_service_image()
