#!/bin/sh
# run on non-interactive logins
# Time-stamp: <2011-09-09 23:55:56 chris>

umask 002

alias ls='ls -1 --color'
alias l='ls'
alias ll='ls -l'
alias lrt='ls -rt'
alias llrt='ls -lrt'
alias la='ls -A'
alias lla='ls -lA'
alias ..='cd ..'
alias -- -='cd -'

alias g=git
alias gi=git
alias gitinfo='git show | head -3 && git remote -v'

alias grep="grep --color"
alias r='stty sane'

h() { history | tail -15; }
s() { [[ $# == 0 ]] && s local || ssh -t $@ "screen -ADR $(whoami)"; }
x() { [[ $# == 0 ]] && s local || ssh -t $@ "screen -Ax  $(whoami)"; }
line()   { sed "$1q;d" $2; }
rmline() { sed -i'' -e "$1d" $2; }
buffer() { (test -t 1 && less -F - || cat) } # stdin to less if terminal
tmpdir() { pushd `mktemp -d -t tmpXXX`; }
mkcd()   { mkdir $@ && cd $@; }
calc()   { echo "$@" | bc -l; }
E() { emacs -bg black -fg white $@; }
e() { emacsclient -t $@; }
mdb() { matlab -Dgdb; }
c() { cal -3 $@; }
dos2unix() { sed 's/\o15//g' -i $@; }

matrix() { tr -c "[:digit:]" " " < /dev/urandom | dd cbs=$COLUMNS conv=unblock | GREP_COLOR="1;32" grep --color "[^ ]"; }

export LESS="-irmXF"
export LESSOPEN="|lesspipe.sh %s"  # special less file hooks
export LESSCLOSE=

alias m="matlab -nojvm -nosplash"
alias M="matlab -nodesktop -nosplash"

if [[ `uname -m` == x86_64 ]]; then ARCH=64; fi
if [[ `uname` == Darwin ]]; then
  VERSION=`sw_vers -productVersion`
  if [[ ${VERSION} =~ 10\.6.* ]]; then # SnowLeopard
    export MACI64=1
  else
    export MACI64=0
  fi
  alias ls='ls -1G' # Darwin hates '--color' parameter
fi

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "$debian_chroot" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
force_color_prompt=yes

# timestap history
export HISTFILE=~/.bash_history
echo "# $(date)" >>$HISTFILE
export HISTFILESIZE=10000
export HISTSIZE=10000
export HISTCONTROL=ignoreboth # Don't store duplicate adjacent items in the history 
shopt -s histappend
export PROMPT_COMMAND="history -a && history -r" # each cmd updates hist

# from github.com/huyng/bashmarks
touch ~/.bookmarks
. ~/.bookmarks
function mark {  # mark current directory
    mv ~/.bookmarks /tmp
    grep -v "export DIR_$1=" /tmp/.bookmarks >~/.bookmarks
    echo "export DIR_$1=$PWD" >>~/.bookmarks
}

function j {  # jump to bookmark
   . ~/.bookmarks
   cd $(eval $(echo echo $(echo \$DIR_$1)))
}

function list {  # list bookmarks (with dirname)
   . ~/.bookmarks
   env | grep "^DIR_" | cut -c5- | grep "^.*="
}
function _list {  # list bookmarks (without dirname)
   . ~/.bookmarks
   env | grep "^DIR_" | cut -c5- | grep "^.*=" | cut -f1 -d "="
}

function _jump { # completion command for jump
    local curw
    COMPREPLY=()
    curw=${COMP_WORDS[COMP_CWORD]}
    COMPREPLY=($(compgen -W '`_list`' -- $curw))
    return 0
}
complete -F _jump j
shopt -s progcomp

if [ -r /etc/bash_completion.d/git ]; then
  . /etc/bash_completion.d/git
  #export GIT_PS1_SHOWDIRTYSTATE=1
  complete -o bashdefault -o default -o nospace -F _git g # alias g=git
else
  function __git_ps1() { true; }  # define dummy
fi
export PS1='\[\e[33m\]\h\[\e[0m\].\[\033[32m\]\W\[\033[0m\]$(__git_ps1 "{%s}") \$ '

set visual-bell none

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
    . /etc/bash_completion
fi

# setup PATH (top of list is highest precedence)
[[ `uname` =~ CYGWIN.* ]] || PATH=   # windows already set PATH
for p in \
    $HOME/.bin \
    /Applications/Aquamacs.app/Contents/MacOS/bin/ \
    /usr/local/bin \
    /usr/local/sbin \
    /opt/local/bin \
    /opt/local/sbin \
    /usr/bin \
    /usr/sbin \
    /bin \
    /sbin \
    /usr/X/bin \
    /usr/X11R6/bin \
    /usr/local/matlab/bin \
    /usr/local/cuda/bin \
    ; do
  [ -x $p ] && PATH=$PATH:$p
done
unset p
export PATH=${PATH##:}

# more paths
export LD_LIBRARY_PATH=/usr/local/cuda/lib/:$LD_LIBRARY_PATH
export MANPATH=/usr/share/man:/usr/local/man:/opt/local/man:/usr/local/cuda/man
export INFOPATH=/usr/local/info:/usr/share/info:/opt/local/info

# editors
export PAGER=less
export VISUAL=nano
export EDITOR=nano
export ALTERNATE_EDITOR=nano

# android paths
export NDKROOT=/home/chris/workspace/android-ndk-r4-crystax
export NDK_ROOT=$NDKROOT
export ANDROID_NDK=$NDKROOT
export ANDROID_NDK_ROOT=$NDKROOT
export ANDROID_SDK=/home/chris/workspace/android-sdk-linux_x86
export PATH=$PATH:$ANDROID_SDK/tools:$ANDROID_SDK/platform-tools:$ANDROID_NDK/

#up-arrow-history
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'

# GIT PS1 and auto complete
if [ -r /etc/bash_completion.d/git ]; then
    . /etc/bash_completion.d/git
else
    function __git_ps1() { true; }
fi

