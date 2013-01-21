;
; red blue channel swap
;
; by Stu Fisher http://q3f.org
; modified by Chris McClanahan http://mcclanahoochie.com
;

(define (red_blue_swap filename)
  (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
         (drawable (car (gimp-image-get-active-layer image))))    
    (plug-in-colors-channel-mixer RUN-NONINTERACTIVE image drawable FALSE 0 0 1 0 1 0 1 0 0)
    (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
    (gimp-image-delete image)))
