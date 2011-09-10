%Function to read a video from the webcam in color or greylevel
%using opencv
%INPUTS
%	frames    -     frames to read
%	colorflag - 	if 1 reads 3 levels if 1 converts directly to greylevel
%OUTPUT
%	V         -	HxWx(3 or 1)x(FRAMES) array containing image.
%V = read_from_camera(frames,color_flag);
