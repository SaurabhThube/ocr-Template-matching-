# ocr-Template-matching-
Template matching ocr using scanlines and templates. Accuracy more than 80%. Need to improve accuracy and small character recognition more precisely
templates are provided and can be changed by contributors.
Any changes that contribute to improving accuracy of this project is always a welcome! :)

Project Brief:
1) Project uses concept of scanlines and template matching.
2) Its accuracy is more than 80%, also code uses autocorrect module so as to improve accuracy of identified words. sometimes    autocorrect module might deviate from original word.
3) Code creates a matrix of given image and looks for first black pixel and then for complete line with no black pixel. it applies this for both column and rows so as to get a single character from given image. Then it resizes it to templates' size and then invokes cv2 method of 'cv2.TM_SQDIF_NORMED'.
4) Download all the templates provided with code and place code file and all templates in a single folder. place input image also in same folder. 
5) input your sample image through terminal after running code. Suppose sample image is 'sample.jpg' then enter 'sample.jpg' on terminal after executing code file.

     HAPPY CODING!!!
