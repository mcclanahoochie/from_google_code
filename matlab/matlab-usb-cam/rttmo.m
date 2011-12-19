   % Copyright [2011] [Chris McClanahan]

   % Licensed under the Apache License, Version 2.0 (the "License");
   % you may not use this file except in compliance with the License.
   % You may obtain a copy of the License at

   %     http://www.apache.org/licenses/LICENSE-2.0

   % Unless required by applicable law or agreed to in writing, software
   % distributed under the License is distributed on an "AS IS" BASIS,
   % WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   % See the License for the specific language governing permissions and
   % limitations under the License.


close all 
clear all


large = 0;

ii = 10;


while (ii),

    img0 = read_from_camera(3,1);
    img1 = img0(:,:,:,1); 
    img2 = img0(:,:,:,2); 
    img3 = img0(:,:,:,3); 

    figure(1); 
    if large == 1,
	    limg1 = imresize(img1,2);
	    imshow(limg1); 
	else
	    imshow(img1); 
	end
    drawnow;

    figure(2); 
    if large == 1,
	    limg2 = imresize(img2,2);
	    imshow(limg2); 
	else
	    imshow(img2); 
	end
    drawnow;

    figure(3); 
    if large == 1,
	    limg3 = imresize(img3,2);
	    imshow(limg3); 
	else
	    imshow(img3); 
	end
    drawnow;

    hdr = mymakehdr(img0);
    %tone = tonemap(hdr, 'AdjustLightness', [0.1 0.9]);
    tone = tonemap(hdr, 'AdjustLightness', [0.1 0.9], 'AdjustSaturation', 1.5);
    %tone = tonemap(hdr);

    figure(5); 
    if large == 1,
	    ltone = imresize(tone,2);
	    imshow(ltone); 
	else
	    imshow(tone); 
	end
    drawnow;

    pause(.05); 
    ii = ii -1

end

close all
