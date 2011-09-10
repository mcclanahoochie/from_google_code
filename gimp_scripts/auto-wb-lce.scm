; My first script! (modified from an example on gimp.org)
;
;   It applies optionally:
;     -Auto White Balance 
;     -Local Contrast Enhancement 
;     -LOMO
;   effects to an image.
;
; Requires Fx-Foundry (gimp-plugin-registry) 
;   for LCE & LOMO to work
;
; Install to ~/.gimp-2.x/scripts
;
; ~Chris McClanahan

(define (script-fu-auto-wb-lce 
         image
         drawable
         check-awb
         check-lce
         check-lom
         )


; ======================================= ;
; setup
; ======================================= ;

  (let* (
         (work-layer -1)
         ) ; End of variable definitions

    ; Save the current settings
    (gimp-context-push)

    ; Start an undo group so the image can be restored with one undo
    (gimp-image-undo-group-start image)

    ; Clear any selections to avoid execution errors. 
    (gimp-selection-none image)

    ; copy visible into a work-layer
    (set! work-layer (car (gimp-layer-new-from-visible image image "Enhanced") ))
    (gimp-image-add-layer image work-layer 0) 

    ; activate
    (gimp-image-set-active-layer image work-layer) 

; ======================================= ;
; auto white balance
; ======================================= ;

    ; Call a plugin to process the image
    (if (= check-awb TRUE)
        (gimp-levels-stretch work-layer)
        ())
    
    ; activate
    (set! work-layer (car (gimp-image-get-active-layer image)))     
    (gimp-image-set-active-layer image work-layer) 

; ======================================= ;
; local contrast enhancement
; ======================================= ;

    ; Call a plugin to process the image
    (if (= check-lce TRUE)
        (script-fu-LCE 1 work-layer work-layer 45 10)
        ())
    
    ; activate
    (set! work-layer (car (gimp-image-get-active-layer image)))     
    (gimp-image-set-active-layer image work-layer) 

; ======================================= ;
; LOMO script
; ======================================= ;

    ; Call a plugin to process the image
    (if (= check-lom TRUE)
        (script-fu-lomo-effect 1 work-layer work-layer 0 0)
        ())
    
    ; activate
    (set! work-layer (car (gimp-image-get-active-layer image)))     
    (gimp-image-set-active-layer image work-layer) 


; ======================================= ;
; finish
; ======================================= ;

    ; Finish the undo group for the process
    (gimp-image-undo-group-end image)

    ; Display the updated image
    (gimp-displays-flush)

    ; Restore previous settings
    (gimp-context-pop)  

    ) ; End of Let block
  )  ; End of Define block

; ======================================= ;
; dialog setup
; ======================================= ;
(script-fu-register "script-fu-auto-wb-lce"
                    "Auto WB+LCE"
                    "Auto White Balance and Local Contrast Enhancement"
                    "Chris McClanahan"
                    "Chris McClanahan"
                    "2011"
                    "RGB*, GRAY*"
                    SF-IMAGE    "Image"    0
                    SF-DRAWABLE "Drawable" 0
                    SF-TOGGLE   "AWB"      TRUE
                    SF-TOGGLE   "LCE"      TRUE
                    SF-TOGGLE   "LOMO"     FALSE
                    )

(script-fu-menu-register "script-fu-auto-wb-lce"
                         "<Image>/Script-Fu/Color")
