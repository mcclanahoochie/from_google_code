;;; screenserver.el --- Use GNU Emacs in server mode in a screen session
;;; Copyright (C) 2002-2003 Ben Jansens (ben@orodu.net)
;;;
;;; Author: Ben Jansens (ben@orodu.net)
;;;
;;; This file is not part of GNU Emacs.
;;;
;;; This program is free software; you can redistribute it and/or modify
;;; it under the terms of the GNU General Public License as published by
;;; the Free Software Foundation; either version 2, or (at your option)
;;; any later version.
;;;
;;; This program is distributed in the hope that it will be useful,
;;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;;; GNU General Public License for more details.

;;; Description:
;;; This provides a mechanism by which you can run emacs in server mode in a
;;; screen session, When emacs is run, it will automatically switch to the
;;; screen window with emacs server running. When finished in the buffer
;;; (C-x #), this will automatically switch the user back to the screen window
;;; they originated from.
;;;
;;; Installation:
;;; Put the following in your .emacs:
;;;
;;;    (require 'screenserver)
;;;
;;; Make an 'emacs' script in your path (something like ~/bin/emacs) that does
;;; the following:
;;;
;;;    #! /bin/sh
;;;    echo $WINDOW > /tmp/$USER-emacsclient-caller
;;;    screen -r -X select `cat /tmp/$USER-emacsserver-window`
;;;    emacsclient $*
;;;
;;; Make sure this script is in your path before the actual emacs program.
;;; Do 'echo $PATH' to see the order.
;;;
;;; Now start emacs in server mode. A convenient way to do this is in your
;;; ~/.screenrc file, with a line such as:
;;;
;;;    screen -t Emacs 0 /usr/bin/emacs
;;;
;;; Or simply run /path/to/real/emacs in a screen window to initiate the
;;; server.
;;;
;;; That's it! If all went well, when you type 'emacs somefile' in another
;;; screen window, it'll switch to the server, and switch back when you're
;;; done with it (C-x #).

(provide 'screenserver)

(server-start)
(shell-command "echo $WINDOW > /tmp/$USER-emacsserver-window")
(add-hook 'after-init-hook 'server-start)
(add-hook 'server-done-hook
	  (lambda ()
	    (shell-command
	     "screen -r -X select `cat /tmp/$USER-emacsclient-caller`")))

