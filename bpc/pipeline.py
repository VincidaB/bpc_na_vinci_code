import roslibpy
import time

import cv2
import base64

import numpy as np
from cv_bridge import CvBridge
bridge = CvBridge()

# just a test to see if I can subscribe to a topic from a conda env

# subribe to the topic /chatter for now


def pipeline_service():
    # Create a Roslibpy client
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    # Create a service server for the /process_image service
    service = roslibpy.Service(client, '/get_pose_estimates', 'ibpc_interfaces/srv/GetPoseEstimates')

    # Define a callback function to handle service requests
    def handle_request(request, response):
        """
        returns a response of the form `PoseEstimagte[]` in the response of name `pose_estimates`
        """
        print('Received request:', request)         

        # Extract the object_ids, cameras, and photoneo from the request
        object_ids = request['object_ids']
        cameras = request['cameras']
        photoneo = request['photoneo']

        print(f'Object IDs: {object_ids}')


        print('Processed image and sent response.')
        
        # ? that is how we set the response !!!!!
        #response["pose_estimates"] = True 
        return True

    # Advertise the service with the callback function
    service.advertise(handle_request)

    print('Service /get_pose_estimates is now available.')

    # Keep the script running to handle service requests
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # Unadvertise the service and close the client
    service.unadvertise()
    client.terminate()




if __name__ == '__main__':
    

    # start the pipeline service
    pipeline_service()