; === Batch Auto White Balance, Strech Contrast, Stretch HSV ===
;
; By:
;   Chris McClanahan
;
; Modified version of:
;   http://drupalique.com/story/2011-02-02
;   /gimp-batch-image-processing-auto-color-enhance-levels-resize-and-save
;
; Install to (Linux):
;    ~/.gimp-2.x/scripts
; Install to (Mac):
;    ~/Library/Application\ Support/Gimp/scripts/
;
; run with (Linux):
;    gimp -i -b '(batch-auto-levels "*.jpg")' -b '(gimp-quit 0)'
; run with (Mac):
;    /Applications/Gimp.app/Contents/Resources/bin/gimp -b '(batch-auto-levels "*.jpg")' -b '(gimp-quit 0)'

(define (batch-auto-levels pattern)
  ; file list
  (let* ((filelist (cadr (file-glob pattern 1))))
    ; loop all 
    (while (not (null? filelist))
           ; load 
           (let* ((filename (car filelist))
                  (nfilename (string-append "G_" filename))
                  (image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
                  (drawable (car (gimp-image-get-active-layer image))))
             ; process 
             (gimp-levels-stretch drawable)
             (plug-in-c-astretch RUN-NONINTERACTIVE image drawable)
             (plug-in-autostretch-hsv RUN-NONINTERACTIVE image drawable)
             ; save
             (gimp-file-save RUN-NONINTERACTIVE image drawable nfilename nfilename)
             (gimp-image-delete image))
           ; next
           (set! filelist (cdr filelist)))))
