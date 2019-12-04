#---Read.me--------#
1     Run vid2img.py to split videos into images
                            update the video_path as video path
							update the save_path as destination folder for images
	

2    -----Run main.py 
                update path on line 55 and 56 acc to your train and test images folder
				update line 102 acc to the index of path(here i split on index 8 according to my path of images)
				update remaining lines from 141 to line 153 in accordance to the folder where you want to save output
3   -------set batch size 64 first then reduce it if Resource Exausted error occurs				