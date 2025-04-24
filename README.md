# Video upscaler
Context:
The aim of the project was to create a video upscaler that would create new pixels in a mp4 file without usage of any generative AI. We aimed to do this solemnly with the usage of data interpolation, different filters like Bilateral Filter and optical flow.

The programme generates new pixels using an interpolation method called bicubic. After that we apply several different filters to sharpen our new frames. Firstly, the program was not very optimised and processed one second of video in about 10 seconds of real time. So we used the optical flow to determine where the areas of greatest movement were in the video and used more advanced filters there and less advanced filters on the rest of the frame to save on computational time. When we were satisfied with the results, we added a convolutional neural network to further improve our final results.

Results:

https://github.com/user-attachments/assets/a073bcbf-25f6-490d-ac7e-0f3431f07e55

https://github.com/user-attachments/assets/6f79b0b8-56a7-4a9c-9cc6-ee5c1c79a582

