addpath ~/jsrc/jacket/engine/
addpath ~/jsrc/jacket/test/harness    
clear all
close all

% settings
iter = 100;
colr = 1;
norm = 0;

% sobel
h0 =  [-2.0, -1.0,  0.0;
       -1.0,  0.0,  1.0;
       0.0,  1.0,  2.0]; 

% robinson
h1 =  [  0.0,  1.0,  1.0;
         -1.0,  0.0,  1.0;
         -1.0, -1.0,  0.0]; 

% gaussian blur
h2a =  single( [ 0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006 ] );
h2 = h2a'*h2a;

% prepare
avgtime = 0;
he = single(h1);
hs = single(h2);

% main
for ii=[1:iter],

  % get
  img0 = read_from_camera(1,colr);
  g1 = gsingle(img0);
  g4 = g1;
  ss = [];
  maxL = 0;
  
  tic;
    %-------------------------
    
    if colr==0,
      
      % smooth
      g2 = conv2(h2a,h2a,g1,'same');
      % histogram
      % g2 = histeq(g2,hist(g2(:),256));
      % edge
      g4 = abs(conv2(g2,h0 ,'same'));
      % normalize to [0-255]
      if (norm==1),
        imin = min(g4(:));
        imax = max(g4(:));
        irng = imax - imin;
        g4 = 255 * (g4 - imin) / irng;
      else
        g4 = adjcontrast(g4,10,0.5);
        g4 = g4*255;
      end
      
    else
      
      % channel split
      rr = g1(:,:,1);
      gg = g1(:,:,2);
      bb = g1(:,:,3);
      if (norm==1),
        % histogram
        rr = histeq(rr,hist(rr(:),256));
        gg = histeq(gg,hist(gg(:),256));
        bb = histeq(bb,hist(bb(:),256));
      else
        % contrast
        rr = adjcontrast(rr,8,0.6)*255;
        gg = adjcontrast(gg,8,0.6)*255;
        bb = adjcontrast(bb,8,0.6)*255;
      end
      % thresh
      tt = (rr > 210) & (gg < 190) & (bb < 190);
      tt = bwmorph(tt,'open');
      tt = bwmorph(tt,'open');
      tt = bwmorph(tt,'open');
      tt = bwmorph(tt,'open');
      L = bwlabel(tt,8);
      maxL = max(L(:));
      ss = regionprops(L, 'Centroid');
      
      % merge
      sz=size(rr);
      g3=gzeros(sz(1),sz(2),3);
      g3(:,:,1) = L*10;
      g3(:,:,2) = L*10;
      g3(:,:,3) = L*10;
      % g3(:,:,1) = tt*128;
      % g3(:,:,2) = tt*128;
      % g3(:,:,3) = tt*128;
      % g3(:,:,1) = rr;
      % g3(:,:,2) = gg;
      % g3(:,:,3) = bb;
      g4=g3;
    end
    

    %-------------------------
    avgtime = avgtime + toc;

    % show orig
    img0 = cast(single(img0),'uint8');
    figure(2); colormap('gray');
    image(img0);
    drawnow;
    
    % show proc
    img1 = cast(single(g4),'uint8');
    figure(3); colormap('jet');
    image(img1);
      
    sz=size(img1);
    maxL
    hold on
      for k = 1:maxL,
        c = double(ss(:,k));
%        c = double(ss(k).Centroid);
        sprintf('%d', k);
        text(c(1), c(2), sprintf('%d', k),'EdgeColor','red');
      end
    hold off
    drawnow;


    % loop
    ii
    pause(.05); 
    
end

% timing
avgtime = avgtime / iter
%close all
