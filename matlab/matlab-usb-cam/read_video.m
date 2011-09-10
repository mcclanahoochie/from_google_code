%Function to read a compressed video in color or greylevel
%using opencv
%INPUTS
%	filename - 	file to read from
%	colorflag - 	if 1 reads 3 levels if 1 converts directly to greylevel
%	first_frame - 	[optional] first frame to read from
%	last_frame  -	[optional] last frame (needed if first frame is specified)
%OUTPUT
%	V                -	HxWx(3 or 1)x(FRAMES) array containing image.
%V = read_video('filename',color_flag,first_frame,last_frame);
