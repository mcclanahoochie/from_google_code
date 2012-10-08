;; Time-stamp: <2012-10-08 09:40:00 chris>
(setq
 user-full-name "Chris McClanahan"
 user-mail-address "chris.mcclanahan@accelereyes.com"
 mail-default-reply-to user-mail-address)
(add-to-list 'load-path (expand-file-name "~/.elisp/"))
(add-to-list 'load-path "/usr/local/share/emacs/site-lisp")

;; debug -- keep near top
(setq
 debug-on-error nil                     ; enter debugger on errors
 stack-trace-on-error nil               ; show backtrace of error on debug
 debug-on-quit nil                      ; don't debug when C-g is hit
 debug-on-signal nil                    ; debug any/every error
 )

(setq-default
 auto-fill-function nil
 indent-tabs-mode nil)
(global-font-lock-mode t)
;;(setq font-lock-maximum-decoration t)
(setq
 transient-mark-mode t
 show-trailing-whitespace t
 colon-double-space t
 pop-up-windows t
 visible-bell t
 visual-bell t                          ; no beeping, just flash screen
 make-backup-files nil                  ; no backup files
 default-major-mode 'text-mode
 inhibit-startup-message nil
 message-log-max t
 search-highlight t
 query-replace-highlight t
 indicate-buffer-boundaries t
 indicate-empty-lines t
 kill-whole-line t
 garbage-collection-messages nil        ; annoying
 require-final-newline t                ; always ensure newline before EOF
 next-line-add-newlines nil             ; no newlines when scrolling down
 scroll-step 1                          ; scroll one line at time
 scroll-preserve-screen-position t      ; keep position when scroll
 scroll-margin 0                        ; lines point remains from edges
 next-screen-context-lines 0            ; scroll full screen like less
 case-fold-search t                     ; case insensitive search
 ispell-silently-savep t                ; save personal dictionary sans conf
 revert-without-query '("\\.log$")
 diff-default-read-only t
 diff-switches "-uw" ; 'c' context, 'u' - universal (+/-)
 )

(setq completion-ignore-case t                 ; ignore case when completing...
      read-file-name-completion-ignore-case t) ; ...filenames too

(setq inhibit-startup-message t          ; don't show ...
  inhibit-startup-echo-area-message t)   ; ... startup messages

(defadvice kill-ring-save (before slick-copy activate compile) "When called
  interactively with no active region, copy a single line instead."
  (interactive (if mark-active (list (region-beginning) (region-end))
                 (message "Copied line")
                 (list (line-beginning-position) (line-beginning-position 2)))))

(defadvice kill-region (before slick-cut activate compile)
  "When called interactively with no active region, kill a single line instead."
  (interactive
    (if mark-active (list (region-beginning) (region-end))
      (list (line-beginning-position)
        (line-beginning-position 2)))))

;(require 'undo-tree)
;(global-undo-tree-mode)

(when (boundp 'aquamacs-version)
  (tabbar-mode 0)
  (aquamacs-autoface-mode 0)
  (setq
   aquamacs-styles-mode nil
   one-buffer-one-frame-mode nil
   aquamacs-save-options-on-quit nil
   mac-command-modifier 'alt
   mac-option-modifier 'meta))

(setq
   x-select-enable-clipboard 't
   smart-frame-positioning-mode nil
   default-frame-alist '((menu-bar-lines . 0)
                         (foreground-color . "white")
                         (background-color . "black")
                         (cursor-color . "Red")
                         (vertical-scroll-bars)
                         (tool-bar-lines . 0)))
(blink-cursor-mode -1)

;; Utility functions
(defun foreach (function sequence)
  "Replaces each element in SEQUENCE with result of calling FUNCTION on it.
Returns reference to modified sequence."
  (defun foreach-h (function sequence)
    (if sequence
        (progn
          (setcar sequence (funcall function (car sequence)))
          (foreach function (cdr sequence)))))
  (foreach-h function sequence)
  sequence)
(defun c-dn-r (sequence n)
  "Returns N number of cdr's on SEQUENCE.  If N is zero, return SEQUENCE."
  (if (eq n 0)
      sequence
    (c-dn-r (cdr sequence) (- n 1))))
;; from Szilagyi via Olin Shivers
(defun find-first (p l)
  "Return the first element of the list L satisfying the predicate P."
  (cond
   ((null l) nil)
   ((funcall p (car l)) (car l))
   (t (find-first p (cdr l)))))

(defadvice switch-to-buffer (around confirm-non-existing-buffers activate)
  "Switch to non-existing buffers only upon confirmation."
  (interactive "BSwitch to buffer: ")
  (if (or (get-buffer (ad-get-arg 0))
          (y-or-n-p (format "`%s' does not exist, create? " (ad-get-arg
                                                             0))))
      ad-do-it))
(defun switch-to-previous-buffer ()
  (interactive)
  (switch-to-buffer (other-buffer)))

(global-set-key "\M-a" 'execute-extended-command) ; less stress on fingers
(global-set-key "\C-x\C-l" 'switch-to-previous-buffer)
(global-set-key "\C-x\C-o" 'other-window)

(require 'cl)
(require 'buffer-stack)
(global-set-key "\C-x\C-j" 'buffer-stack-up)
(global-set-key "\C-x\C-k" 'buffer-stack-down)

;; move directly between windows
(global-set-key "\M-j" 'windmove-down)
(global-set-key "\M-k" 'windmove-up)

;; diminish -- remove clutter of minor modes, abbreviate them on modeline
(require 'diminish)
(eval-after-load "eldoc" '(diminish 'eldoc-mode))
(eval-after-load "refill" '(diminish 'refill-mode "R"))
(eval-after-load "abbrev" '(diminish 'abbrev-mode))
(eval-after-load "undo-tree" '(diminish 'undo-tree-mode))

(add-to-list 'completion-ignored-extensions ".o")
(add-to-list 'completion-ignored-extensions ".d")


;;(require 'tbemail)

;; Tramp
(require 'tramp)
(setq
 tramp-default-method "ssh"
 tramp-verbose 9)

;; sh mode
(defun my-sh-mode ()
  "My shell mode settings."
  (local-set-key "\C-c\C-c" 'compile)
  (local-set-key "\C-cc" 'comment-region))
(add-hook 'sh-mode-hook 'my-sh-mode)
(add-to-list 'auto-mode-alist '("\\.conf$" . sh-mode))



;; eshell, pcomplete
(unless (or (fboundp 'eshell)           ; unless already bound
            (load "eshell-auto" t))     ; or attempt to load works
  (add-to-list 'load-path "~/.elisp/eshell") ; redefine load-path...
  (load "eshell-auto"))                      ; ...and try again
(unless (or (fboundp 'pcomplete)
            (load "pcmpl-auto" t))
  (add-to-list 'load-path "~/.elisp/pcomplete")
  (load "pcmpl-auto"))


;; from http://www.emacswiki.org/cgi-bin/wiki?EshellFunctions
(defun eshell-maybe-bol ()
  (interactive)
  (let ((p (point)))
    (eshell-bol)
    (if (= p (point))
        (beginning-of-line))))
(defun eshell/emacs (&rest args)
  "Open a file in emacs.  Accepts wildcards."
  (if (null args)
      (bury-buffer)
    (mapc #'find-file (mapcar #'expand-file-name args))))
(defun eshell/ec (&rest args)
  "Use `compile' to do background makes."
  (if (eshell-interactive-output-p)
      (let ((compilation-process-setup-function
             (list 'lambda nil
                   (list 'setq 'process-environment
                         (list 'quote (eshell-copy-environment))))))
        (compile (eshell-flatten-and-stringify args))
        (pop-to-buffer compilation-last-buffer))
    (throw 'eshell-replace-command
           (let ((l (eshell-stringify-list (eshell-flatten-list args))))
             (eshell-parse-command (car l) (cdr l))))))
(put 'eshell/ec 'eshell-no-numeric-conversions t)
(defun my-eshell-setup ()
  "My eshell and pcomplete setup.  I had to put these in this hook because
If I just set them globally, they get overriden when eshell starts.  It has
something to do with 'defun which I obviously don't know much about."
  (setq
   eshell-modify-global-environment t   ; export affects emacs
   eshell-ask-to-save-history 'always   ; save history silently
   eshell-cd-shows-directory nil        ; print dir after changing into
   eshell-default-target-is-dot t       ; default dir for cp,mv,ln is '.'
   eshell-prefer-lisp-functions nil
   eshell-plain-grep-behavior nil
   pcomplete-file-ignore nil            ; don't exclude any file completions
   pcomplete-dir-ignore nil             ; don't exclude any dir completions
   eshell-prefer-to-shell t             ; use eshell for M-!
   eshell-history-size 500
   eshell-scroll-show-maximum-output t  ; show max output
   eshell-scroll-to-bottom-on-output nil  ; don't scroll while browsing
   eshell-scroll-to-bottom-on-input t   ; scroll as soon as type
   eshell-output-filter-functions '(eshell-handle-control-codes
                                    eshell-watch-for-password-prompt
                                    eshell-postoutput-scroll-to-bottom
;;                                     comint-show-maximum-output
                                    )
   pcomplete-expand-before-complete nil ; expand as far as possible
   pcomplete-autolist nil               ; 1st TAB partial complete, 2nd list
   pcomplete-cycle-completions nil      ; always show list, no cycle
   pcomplete-cycle-cutoff-length 0      ; no exceptions to cycling
   pcomplete-use-paring nil             ; don't exclude any completions
   pcomplete-restore-window-delay 0
   pcmpl-gnu-tarfile-regexp
   "\\.t\\(ar\\(\\.\\(gz\\|bz2\\|Z\\)\\)?\\|gz\\|a[zZ]\\|z2\\)\\'"
   eshell-tar-regexp pcmpl-gnu-tarfile-regexp
   eshell-show-lisp-completions t
   eshell-modules-list   '(eshell-alias
                           eshell-basic
                           eshell-cmpl
                           eshell-dirs
                           eshell-glob
                           eshell-hist
                           eshell-ls
                           eshell-prompt
                           eshell-unix
                           eshell-smart
                           eshell-term
                           eshell-xtra)
   eshell-visual-commands '("vi"
                            "ssh"
                            "vim"
                            "screen"
                            "python2.6"
                            "top"
                            "gdb"
                            "matlab"
                            "matlab74"
                            "less"
                            "ccmake"
                            "ipython2.6")
   )
  (local-set-key "\C-a" 'eshell-maybe-bol)
  (local-set-key "\C-ce" 'eshell-show-maximum-output)
  (local-set-key "\C-xw" 'w3m)
  (local-set-key "\C-xf" 'w3m-search)
  (local-set-key "\C-x\C-h" 'eshell)
  (local-set-key "\C-ct" 'matlab-shell)
  )
(add-hook 'eshell-mode-hook 'my-eshell-setup)
(global-set-key "\C-x\C-h" 'eshell)


;(load "ange-ftp")

;; menu/tool/scroll bars
(menu-bar-mode -1)
(if (fboundp 'tool-bar-mode) (tool-bar-mode -1))
(if (fboundp 'scroll-bar-mode) (scroll-bar-mode -1))
(if window-system (mwheel-install))     ; mouse scroll wheel works
(setq
 display-time-day-and-date t
 display-time-mail-file t               ; disable mail
 display-time-format "%k:%M [%a]"
 display-time-string-forms '((format-time-string display-time-format now))
 size-indicate-mode t
 column-number-mode t
 line-number-mode t)
(display-time)
(setq
 calendar-week-start-day 1
 calendar-setup 'calendar-only
 calendar-remove-frame-by-deleting t)

;; imenu
;; (require 'imenu)
;; (setq
;;  imenu-sort-function 'imenu--sort-by-name
;;  imenu-always-use-completion-buffer-p t)

(global-set-key (kbd "ESC RET") 'hippie-expand)
(global-set-key "\M-h" 'backward-kill-word)
(global-set-key "\C-cc" 'comment-region)
(global-set-key "\C-c\C-c" 'compile)
(global-set-key "\C-c\C-r" 'refill-mode)
(global-set-key "\C-c\C-v" 'view-mode)
(global-set-key "\C-x\C-v" 'view-file)
(define-key ctl-x-4-map "\C-v"'view-file-other-window)
(define-key ctl-x-5-map "\C-v" 'view-file-other-frame)
(global-set-key (kbd "ESC i")
                '(lambda ()
                   "Insert the output of a shell command"
                   (interactive)
                   (shell-command (read-string "Shell command to insert: ")
                                  t)
                   (exchange-point-and-mark))) ; move to after inserted text
(fset 'view 'view-mode)                        ; instead of 'defalias
(defun my-view-mode-setup ()
  "My settings for view-mode."
  (define-key view-mode-map "b" 'View-scroll-page-backward) ; ~less
  (define-key view-mode-map "k" 'View-scroll-line-backward) ; ~less
  (define-key view-mode-map "j" 'View-scroll-line-forward)  ; ~less
  (define-key view-mode-map (kbd "<up>") 'View-scroll-line-backward)
  (define-key view-mode-map (kbd "<down>") 'View-scroll-line-forward)
  (define-key view-mode-map "G" 'end-of-buffer)       ; ~less
  (define-key view-mode-map "Q" 'View-leave)
  (define-key view-mode-map "q" 'View-kill-and-leave)
  )
(add-hook 'view-mode-hook 'my-view-mode-setup)


;; Man mode
(setq
 Man-notify-method 'pushy)
(defun my-Man-mode ()
  "My Man-mode settings"
  (view-mode 1)
  (define-key view-mode-map "q" 'Man-quit)
  )
(add-hook 'Man-mode-hook 'my-Man-mode)

;; auto-save
(setq
 auto-save-interval 0    ; don't do based on keypresses
 auto-save-timeout 120)  ; only do when idle for 2 min
;; move autosave listings to /tmp -- from Olin Shivers
(condition-case ()
    (let ((d "/tmp/esave/"))
      (if (not (file-directory-p d t))
          (make-directory d t))
      (setq auto-save-list-file-prefix (concat d "es.")))
  (error ()))

;; misc setup
(fset 'yes-or-no-p 'y-or-n-p)           ; all yes-or-no prompts become y-or-n
(define-key query-replace-map (kbd "RET") 'act)

(add-to-list 'completion-ignored-extensions ".pdf")
(add-to-list 'completion-ignored-extensions ".ps")
(add-to-list 'completion-ignored-extensions ".dvi")

(autoload 'pack-windows "pack-windows" nil t) ; equalize windows
(require 'fscroll)                  ; scroll fully to first or last characters

(defun swap-windows ()
  "If you have 2 windows, it swaps them."
  (interactive)
  (cond
   ((not (= (count-windows) 2))
    (message "You need exactly 2 windows to do this."))
   (t
    (let* ((w1 (first (window-list)))
	   (w2 (second (window-list)))
	   (b1 (window-buffer w1))
	   (b2 (window-buffer w2))
	   (s1 (window-start w1))
	   (s2 (window-start w2)))
      (set-window-buffer w1 b2)
      (set-window-buffer w2 b1)
      (set-window-start w1 s2)
      (set-window-start w2 s1)))))

(setq time-stamp-active t)
(add-hook 'write-file-hooks 'time-stamp)


(defun remove-trailing-whitespace ()
  "Remove whitespace at end of lines throughout entire buffer."
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (while (search-forward "[ \t]+$" nil t
                           (replace-match "" nil t)))))
;; (add-hook 'before-save-hook 'delete-trailing-whitespace)

;; print function documentation in minibuffer as you type
(autoload 'turn-on-eldoc-mode "eldoc" nil t)
(add-hook 'emacs-lisp-mode-hook 'turn-on-eldoc-mode)
(add-hook 'lisp-interaction-mode-hook 'turn-on-eldoc-mode)
(add-hook 'ielm-mode-hook 'turn-on-eldoc-mode)

(load "flash-paren")
(flash-paren-mode 1)

(load "rsz-mini")       ; resize minibuffer dynamically to accomodate contents
(setq resize-minibuffer-mode t)

(defun resize-window ()
  "Resize the window interactively"
  (interactive)
  (if (one-window-p) (error "Cannot resize sole window"))
  (let (c)
    (catch 'done
      (while t
        (message "e enlarge vert,   s shrink vert,   else finish")
        (setq c (read-char))
        (cond
         ((= c ?e) (enlarge-window 1))
         ((= c ?s) (enlarge-window -1))
         (t (throw 'done t)))))         ; exit catch
    (message "Done.")))

(defun my-gen-seq (start stop &optional inc)
  "Generate list of numbers from START to STOP, optional increments of INC."
  (let ((ret) '()
        (inc (or inc 1)))
    (while (<= start stop)
      (setq
       ret (append ret (list start))
       start (+ start inc)))
    ret))



;; standard text editing setup
(if (not (fboundp 'caddr))
    (defun caddr (lst) (car (c-dn-r lst 2))))
(if (not (fboundp 'cadddr))
    (defun cadddr (lst) (car (c-dn-r lst 3))))
(if (not (fboundp 'point-at-bol))
    (defun point-at-bol ()
      "Return point at beginning of line."
      (let (bol)
        (save-excursion
          (move-to-column 0)
          (setq bol (point)))
        bol)))
(if (not (fboundp 'make-extent))
    (defun make-extent (beg end)
      (make-overlay beg end)))
(if (not (fboundp 'set-extent-properties))
    (defun set-extent-properties (x y)
      nil))

(defadvice refill-mode (after
                        my-refill-mode
                        activate)
  "Turn back on auto-fill when refill-mode exists."
  (if (not refill-mode)
      (turn-on-auto-fill)))

(defun my-outline-mode ()
  "My outline-mode setup"
  (local-set-key "\C-x\C-h" 'eshell))
(add-hook 'outline-mode-hook 'my-outline-mode)

(defun my-text-mode-settings ()
  "My personal preferences for editing plain-text files."
  (delete-selection-mode t)
  ;; normal text settings
  (local-set-key (kbd "C-c C-v") 'view-mode)
  (local-set-key "\C-c\C-r" 'refill-mode)
  (modify-syntax-entry ?\$ "." text-mode-syntax-table)
  ;; (hl-line-mode 1)
  (setq
   fill-column 90
   sentence-end-double-space t
   colon-double-space t
   tab-width 4
   word-wrap nil
   fill-prefix nil
   tab-stop-list (funcall 'my-gen-seq 0 120 tab-width)
   indent-tabs-mode nil))
(add-hook 'fundamental-mode-hook 'my-text-mode-settings)
(add-hook 'text-mode-hook 'my-text-mode-settings)
(add-hook 'emacs-lisp-mode-hook 'my-text-mode-settings)
(add-hook 'LaTeX-mode-hook 'my-text-mode-settings)

;; latex
(setq TeX-auto-save t
      TeX-parse-self t
      TeX-save-query nil)
(defun my-LaTeX-mode ()
  "My LaTeX mode settings."
  (setq ispell-parser 'tex)
  (define-key LaTeX-mode-map "\C-c\C-j" 'flyspell-check-previous-highlighted-word)
  (define-key LaTeX-mode-map "\C-x\C-h" 'eshell)
  (font-lock-add-keywords
      'LaTeX-mode
      '(("\\<\\(autoref\\)" 1 font-lock-warning-face t)))

  )
(add-hook 'LaTeX-mode-hook 'my-LaTeX-mode)
(mapcar '(lambda (ext)
           (add-to-list 'completion-ignored-extensions ext))
        '(".aux" ".log" ".brf" ".out" ".snm"
          ".toc" ".nav" ".bbl" ".blg" ".idx"
          ".rel"))
(global-set-key (kbd "C-h") 'backward-delete-char-untabify)

(defun del2white ()
  "Deletes whitespace up to the next non-whitespace char."
  (interactive)
  (save-excursion
    (if (re-search-forward "[ \t\r\n]*[^ \t\r\n]" nil t)
        (delete-region (match-beginning 0) (- (point) 1)))))

;; flyspell
(autoload 'flyspell-mode "flyspell" "On-the-fly spelling checker." t)
(eval-after-load "flyspell" '(diminish 'flyspell-mode))
(setq
 flyspell-sort-corrections nil
 flyspell-highlight-properties t)
(global-set-key "\C-c\C-j" 'flyspell-check-previous-highlighted-word)
(add-hook 'text-mode-hook 'flyspell-mode)

(defun complete-or-indent ()
  "Complete word if at end of one, else indent [emacs-wiki]."
  (interactive)
  (if (looking-at "\\>")
      (dabbrev-expand nil)
    (indent-for-tab-command)))

(setq
 compile-command "make -j5"
 compilation-scroll-output t
 compilation-read-command nil
 compilation-ask-about-save nil
 compilation-window-height nil
 compilation-process-setup-function nil)
(autoload 'compilation-always-kill-mode "compilation-always-kill" nil t)
(eval-after-load "compile" '(compilation-always-kill-mode 1))
(defun my-compilation-mode ()
  "My compilation-mode settings."
  (local-set-key "\C-c\C-c" 'compile))
(add-hook 'compilation-mode-hook 'my-compilation-mode)
(defun my-compilation-finish (buffer status)
  "Close compilation `buffer' if `status' is \"finished\"."
  (when (and (equal major-mode 'c-mode)
             (equal status "finished\n"))
    (delete-windows-on buffer)
    (bury-buffer buffer))
  nil)
(setq compilation-finish-function 'my-compilation-finish)
(defun my-compile ()
  "If no Makefile, run make with target of filename without extension."
  (unless (or (file-readable-p "Makefile") (file-readable-p "makefile"))
    (make-local-variable 'compile-command)
    (setq compile-command (concat compile-command
                                  (file-name-sans-extension
                                   buffer-file-name))
          compilation-scroll-output t))
  (compile))
(setq
 gdb-show-main t
 gdb-use-inferior-io-buffer t)
(defun my-gud-mode ()
  "My gud-mode settings."
  (local-set-key "\C-x\C-h" 'eshell))
(add-hook 'gud-mode-hook 'my-gud-mode)

(defun my-c-mode ()
  "C mode with adjusted defaults."
  (c-mode)
  (my-algol-mode))
(defun my-c++-mode ()
  "C++ mode with adjusted defaults."
  (c++-mode)
  (my-algol-mode))

(autoload 'js2-mode "js2" nil t)
(add-to-list 'auto-mode-alist '("\\.js$" . js2-mode))

(defun my-java-mode ()
  "Java mode with adjusted defaults."
  (c-mode)
  (my-algol-mode))
(add-hook 'java-mode-hook 'my-java-mode)
(add-to-list 'auto-mode-alist '("\\.shimp\\'" . java-mode))

(defun my-algol-mode ()
  "Mode for Algol-like languages."
  (c-set-style "java")
  (local-set-key "\C-x\C-n" 'next-error)
  (local-set-key "\C-x\C-p" 'previous-error)
  (local-set-key "\M-j" 'windmove-down)
  (local-set-key "\M-k" 'windmove-up))

(add-to-list 'auto-mode-alist '("\\.[chx]$"  . my-c-mode))
(add-to-list 'auto-mode-alist '("\\.[tc]xx$" . my-c++-mode))
(add-to-list 'auto-mode-alist '("\\.[ch]$"   . my-c-mode))


(defun my-c-mode-common-hook ()
  ;; add my personal style and set it for the current buffer
  ;;   (c-add-style "PERSONAL" my-c-style t)
  (c-set-style "linux")
  (setq c-basic-offset 4)
  (which-func-mode t)
  ;; (font-lock-add-keywords nil
  ;;                         '(("\\<\\(FIXME\\|HACK\\|TODO\\|XXX+\\|BUG\\):?"
  ;;                            1 font-lock-warning-face prepend)))
  (setq
   show-trailing-whitespace t
   default-tab-width 4
   tab-width 4
   indent-tabs-mode nil)
  ;; auto-newline and hungry-delete
  (c-toggle-auto-hungry-state -1)
  (c-toggle-auto-state -1)
  ;; keybindings for all supported languages.  We can put these in
  ;; c-mode-base-map because c-mode-map, c++-mode-map, objc-mode-map,
  ;; java-mode-map, idl-mode-map, and pike-mode-map inherit from it.
  (mapcar (lambda (mode-map)
            (define-key mode-map (kbd "<tab>") 'complete-or-indent)
            (define-key mode-map "\C-cc" 'comment-region)
            (define-key mode-map "\C-c\C-c" 'compile)
            (define-key mode-map "\C-x\C-h" 'eshell)
            (define-key mode-map "\C-ct" 'matlab-shell)
            )
          (list c-mode-map c++-mode-map)))

(add-hook 'c-mode-hook 'my-c-mode-common-hook)
(add-hook 'c++-mode-hook 'my-c-mode-common-hook)
(add-to-list 'auto-mode-alist '("\\.txx$" . c++-mode))

(when (require 'rainbow-delimiters nil 'noerror)
  (add-hook 'c-mode-hook 'rainbow-delimiters-mode)
  (add-hook 'c++-mode-hook 'rainbow-delimiters-mode)
  (add-hook 'emacs-lisp-mode-hook 'rainbow-delimiters-mode))

;; makefile
(add-to-list 'auto-mode-alist '("\\.mak$" . makefile-mode))
(add-to-list 'auto-mode-alist '("Makefile\\." . makefile-mode))
(defun my-makefile-mode ()
  (define-key makefile-mode-map "\C-c\C-c" 'compile))
(add-hook 'makefile-mode-hook 'my-makefile-mode)
;; cmake
(autoload 'cmake-mode "cmake-mode" nil t)
(add-to-list 'auto-mode-alist '("CMakeLists\\.txt$" . cmake-mode))
(add-to-list 'auto-mode-alist '("\\.cmake$" . cmake-mode))


;; uniquify
(require 'uniquify)
(setq
 uniquify-buffer-name-style 'post-forward-angle-brackets)

;; octave-mode
(autoload 'octave-mode "octave-mod" nil t)
(autoload 'run-octave "octave-inf" nil t)
;; (add-to-list 'auto-mode-alist '("\\.m$" . octave-mode))
(defun my-octave-mode ()
  "My Octave mode setup"
  (local-set-key "\C-x\C-h" 'eshell)
  (local-set-key "\C-m" 'octave-reindent-then-newline-and-indent)
  )
(add-hook 'octave-mode-hook 'my-octave-mode)

(autoload 'php-mode "php-mode" nil t)
(add-to-list 'auto-mode-alist '("\\.phtml$" . php-mode))
(add-to-list 'auto-mode-alist '("\\.php$"   . php-mode))

;; matlab-mode
(autoload 'matlab-mode "matlab" "Major mode for editing MATLAB code." t)
(add-to-list 'auto-mode-alist '("\\.m$" . matlab-mode))
(defun my-matlab-mode ()
  "My Matlab mode setup"
  (local-set-key "\C-h" 'backward-delete-char-untabify)
  (local-set-key "\C-cc" 'comment-region)
  (setq
   show-trailing-whitespace t
   matlab-auto-fill nil
   matlab-fill-code nil
   matlab-fill-strings nil
   matlab-indent-function t
   matlab-highlight-block-match-flag t
   matlab-verify-on-save-flag nil
   matlab-handle-graphics-list nil
   matlab-show-periodic-code-details-flag nil))
(add-hook 'matlab-mode-hook 'my-matlab-mode)
(mapcar '(lambda (ext)
           (add-to-list 'completion-ignored-extensions ext))
        '(".mexmaci" ".mexmaci64" ".mexglx" ".mexa64" ".mexw32" ".mexw64"))

(defun my-comint-mode ()
  "My comint setup"
  (message "my-comint-mode")
  (setq
   comint-scroll-to-bottom-on-output 'all
   comint-scroll-show-maximum-output 'all))
(add-hook 'comint-mode-hook 'my-comint-mode)

(add-to-list 'auto-mode-alist '("\\.F90$" . fortran-mode))
(add-to-list 'auto-mode-alist '("\\.cu$" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.inl$" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.cl$" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.pde$" . java-mode))


(defun bsd-c-mode-user-setup ()
  "BSD c-mode"
  (interactive)
  (c-set-style "bsd")
  (setq c-basic-offset 4
        c-conditional-key c-C++-conditional-key
        indent-tabs-mode nil
        c-tab-always-indent t)
  (setq c-cleanup-list (append c-cleanup-list (list 'brace-else-brace)))
  (c-set-offset 'arglist-close 0)
  (c-set-offset 'arglist-cont-nonempty 4)
  (c-set-offset 'inline-open 0)
  (c-set-offset 'case-label 0)
  (c-set-offset 'statement-cont 4)
  (c-toggle-auto-state 1))


;; from emacswiki.org
(setq
 my-shebang-patterns '("^#!/usr/.*/perl.*"
                       "^#!/usr/.*/sh"
                       "^#!/usr/.*/bash"
                       "^#!/bin/sh"
                       "^#!/bin/bash"
                       "^#!/usr/bin/awk .*"))
(defun my-make-file-executable-if-script ()
  "Make buffer executable after saving if it is a script matching patterns
in 'my-shebang-patterns."
  (unless (file-executable-p (buffer-file-name))
    (save-excursion
      (goto-char (point-min))
      (foreach '(lambda (x)
                  (if (looking-at x)
                      (shell-command (concat "chmod +x "
                                             (buffer-file-name))))
                  x)                    ; because foreach side-effects list
               my-shebang-patterns))))
(add-hook 'after-save-hook 'my-make-file-executable-if-script)

;; expire (kill) buffers after inactivity
(autoload 'turn-on-tempbuf-mode "tempbuf" "Auto kill buffers." t)
(add-hook 'dired-mode-hook 'turn-on-tempbuf-mode)
(add-hook 'custom-mode-hook 'turn-on-tempbuf-mode)

(put 'downcase-region 'disabled nil)  ;; wtf?

(add-to-list 'auto-mode-alist '("\\.ptx$" . asm-mode))

;; (load "/opt/local/share/emacs/site-lisp/sml-mode/sml-mode-startup")

(defun rename-file-and-buffer (new-name)
  "Renames both current buffer and file it's visiting to NEW-NAME."
  (interactive "sNew name: ")
  (let ((name (buffer-name))
        (filename (buffer-file-name)))
    (if (not filename)
        (message "Buffer '%s' is not visiting a file!" name)
      (if (get-buffer new-name)
          (message "A buffer named '%s' already exists!" new-name)
        (progn
          (rename-file name new-name 1)
          (rename-buffer new-name)
          (set-visited-file-name new-name)
          (set-buffer-modified-p nil))))))
(defun move-buffer-file (dir)
  "Moves both current buffer and file it's visiting to DIR."
  (interactive "DNew directory: ")
  (let* ((name (buffer-name))
         (filename (buffer-file-name))
         (dir
          (if (string-match dir "\\(?:/\\|\\\\)$")
              (substring dir 0 -1) dir))
         (newname (concat dir "/" name)))

    (if (not filename)
        (message "Buffer '%s' is not visiting a file!" name)
      (progn
        (copy-file filename newname 1)
        (delete-file filename)
        (set-visited-file-name newname)
        (set-buffer-modified-p nil)
        t))))


(iswitchb-mode 1)
(setq
 iswitchb-max-to-show 7
 iswitchb-default-method 'samewindow)

;;(require 'mediawiki)
;;(setq mediawiki-site-alist '(("Wikipedia" "http://en.wikipedia.org/w/"
;;                              "" "" "Main Page")
;;                             ("Internal" "http://coors.a/wiki/"
;;                              "" "" "Main Page")
;;                             ("External" "http://wiki.accelereyes.com/wiki/"
;;                              "" "" "Main Page")))


(defun swap-words (a b)
  "Replace all occurances of a with b and vice versa"
  (interactive "*sFirst Swap Word: \nsSecond Swap Word: ")
  (save-excursion
    (while (re-search-forward (concat (regexp-quote a) "\\|" (regexp-quote b)))
      (if (y-or-n-p "Swap?")
          (if (equal (match-string 0) a)
              (replace-match (regexp-quote b))
            (replace-match (regexp-quote a))))
      )))

;; (add-to-list 'load-path (expand-file-name "~/.emacs.d/skype"))
;; (require 'skype)

(remove-hook 'text-mode-hook 'auto-detect-longlines) ;; Aquamacs v1.4

;; ;; crypt
;; (require 'crypt++)
;; (setq
;;  crypt-encryption-type 'bf
;;  crypt-confirm-password t)
;; (mapcar '(lambda (ext)
;;            (modify-coding-system-alist 'file ext 'no-conversion))
;;         '("\\.bz\\'"
;;           "\\.bz2\\'"
;;           "\\.gz\\'"
;;           "\\.bf\\'"
;;           "\\.Z\\'"))


(defun duplicate-current-line ()
  (interactive)
  (beginning-of-line nil)
  (let ((b (point)))
    (end-of-line nil)
    (copy-region-as-kill b (point)))
  (beginning-of-line 2)
  (open-line 1)
  (yank)
  (back-to-indentation))

(global-set-key "\M-o" 'duplicate-current-line)

(defun act-on-buffers (pred action)
  (let ((count 0))
	(dolist(buffer (buffer-list))
	  (when (funcall pred buffer)
		(setq count (1+ count))
		(funcall action buffer)))
	count))

(defun kill-matching-buffers-by (acc)
  "Kill buffers whose name matches the input"
  (let*
	  ((re (read-string (format "kill buffers matching: ")))
	   (match (function
			   (lambda (buf)
				 (string-match re (funcall acc buf)))))
	   (kill #'(lambda (buf) (kill-buffer buf)))
	   (count (act-on-buffers match kill)))

    (message "%d buffer(s) killed" count)))

(defun kill-matching-buffers-by-name ()
  (interactive)
  (kill-matching-buffers-by 'buffer-name))


(defun kill-matching-buffers-by-filename ()
  (interactive)
  (kill-matching-buffers-by #'(lambda (b) (or (buffer-file-name b) ""))))


(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(diff-added ((t (:foreground "#559944"))))
 '(diff-context ((t nil)))
 '(diff-file-header ((((class color) (min-colors 88) (background dark)) (:foreground "RoyalBlue1"))))
 '(diff-function ((t (:foreground "#00bbdd"))))
 '(diff-header ((((class color) (min-colors 88) (background dark)) (:foreground "RoyalBlue1"))))
 '(diff-hunk-header ((t (:foreground "#fbde2d"))))
 '(diff-nonexistent ((t (:inherit diff-file-header :strike-through nil))))
 '(diff-refine-change ((((class color) (min-colors 88) (background dark)) (:background "#182042"))))
 '(diff-removed ((t (:foreground "#de1923"))))
 )

;; (server-start)


;; (add-to-list 'crypt-encryption-alist
;;              ;; Blowfish encryption
;;              (list 'bf
;;                    crypt-encryption-magic-regexp
;;                    crypt-encryption-magic-regexp-inverse
;;                    (or crypt-encryption-file-extension "\\(\\.bf\\)$")
;;                    "openssl" "openssl"
;;                    '("bf" "-salt" "-k")
;;                    '("bf" "-d" "-k")
;;                    "blowfish"
;;                    nil
;;                    t
;;                    nil))


;; ================================================= ;;

;; lua mode
(load-file "~/.elisp/lua-mode.el")
(autoload 'lua-mode "lua-mode" "Lua editing mode." t)
(add-to-list 'auto-mode-alist '("\\.lua$" . lua-mode))
(add-to-list 'interpreter-mode-alist '("lua" . lua-mode))

;; astyle
(defun astyle-this-buffer (pmin pmax)
  (interactive "r")
  (shell-command-on-region pmin pmax
                           "astyle --style=java --indent=spaces=4 --pad-oper --pad-header  --unpad-paren --add-brackets --keep-one-line-blocks --keep-one-line-statements --align-pointer=type "
                           (current-buffer) t
                           (get-buffer-create "*Astyle Errors*") t))

;; ;; color theme
(require 'color-theme)
(setq color-theme-is-global t)
(setq color-theme-is-cumulative t)
(setq color-theme-load-all-themes nil)
(load-file "~/.elisp/.color-theme.el")
(color-theme-dark-laptop)
;; (color-theme-ld-dark)

;; misc
(savehist-mode t)
(show-paren-mode t)
(setq truncate-lines t)

;; font
(set-default-font "-microsoft-Consolas-bold-bold-bold-*-16-*-*-*-m-0-iso10646-1")
;;(set-default-font "-apple-Menlo-medium-medium-medium-*-16-*-*-*-m-0-iso10646-1")
(set-face-attribute 'default nil :height 150)

;; mac
;; (setq mac-option-modifier 'control)
;; (setq mac-command-modifier 'meta)

;; SHIFT-arrow between windows
(when (fboundp 'windmove-default-keybindings)
  (windmove-default-keybindings))

;; window opacity
(set-frame-parameter nil 'alpha 80)

;; ================================================= ;;
