#!/bin/bash

#
#   Copyright [2011] [Chris McClanahan]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Update Fresh Ubuntu 0.4.4
# #  ~ Chris McClanahan
# #  ~ mcclanahoochie.com
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # Start # # # # # # # # # # # # # # # # # # # # # #

# FYI & intro to true/false usage
if true ; then
    echo "===== Update Fresh Ubuntu 0.4.3 ===== ====="
    echo "  Please configure this script first!"
    echo "   1. Toggle true/false for each section as desired."
    echo "   2. Add/remove packages to install sections as desired."
    echo "   3. Enjoy!"
    echo "===== ===== ===== ===== ===== ====="
    exit 0;
fi

# # # # # # # # # # # # Setup # # # # # # # # # # # # # # # # # # # # # #

# Misc
prog=`basename "$0" .sh`
progDir=`dirname "$0"`
echo="echo $prog:"
echo_exec="echo +"
echo_error="echo $prog: Error:"

# temp folder in ~/
dir="$HOME/tmp"
test -d "$dir" || mkdir "$dir"
$echo_exec cd "$dir"
if ! cd "$dir" ; then
    $echo_error "Could not cd to \"$dir\"" >&2
    exit 1
fi

# Run this to go ahead and get sudo access.
echo "# Checking for sudo access... "
sudo ls >/dev/null

# # # # # # # # # # # # sudo access # # # # # # # # # # # # # # # # # # #

if true ; then
    # Disable sudo asking for your password, forever
    echo "# Disabling sudo asking for password..."
    sudo sed -ie '/^%admin/s/) ALL$/) NOPASSWD: ALL/' /etc/sudoers
    ## echo " ===== Done."
fi

# # # # # # # # # # # # Installs 1 # # # # # # # # # # # # # # # # # # #

echo "5" ;

if true ; then
    echo "# Installing various gnome packages..."
    sudo apt-get -y --force-yes install \
    nautilus-wallpaper \
    nautilus-open-terminal \
    nautilus-data \
    nautilus-actions \
    nautilus-script-manager \
    nautilus-gksu \
    ubuntu-restricted-extras \
    compizconfig-settings-manager \
    compiz-fusion-plugins-extra \
    gedit-plugins \
    aptitude
fi

echo "10" ;

if true ; then
    echo "# Installing various utility packages..."
    sudo apt-get -y --force-yes install \
    ntfs-3g \
    ntfs-config \
    preload \
    gparted \
    wally \
    vlc \
    openvpn \
    audacity \
    emacs23 \
    emacs-goodies-el \
    emacs-goodies-extra-el \
    meld \
    gnome-do \
    hal
fi

echo "15" ;

if true ; then
    echo "# Installing webupd8 reccomended packages..."
    sudo apt-get -y --force-yes install \
    gstreamer0.10-ffmpeg \
    gstreamer0.10-plugins-base \
    gstreamer0.10-plugins-bad \
    gstreamer0.10-plugins-ugly \
    gstreamer0.10-plugins-good \
    libdvdnav4 \
    libdvdread4 \
    libmp4v2-0 \
    libxine1-ffmpeg \
    ffmpeg \
    flashplugin-nonfree \
    rar \
    unrar \
    p7zip-full \
    p7zip-rar \
    zip \
    unzip \
    mplayer \
    sun-java6-plugin \
    sun-java6-jre \
    mozilla-plugin-vlc \
    openshot \
    gimp \
    skype
fi

echo "25" ;

if true ; then
    echo "# Installing various programming packages..."
    sudo apt-get -y --force-yes install \
    subversion \
    subversion-tools \
    astyle \
    build-essential \
    automake \
    libtool \
    libglut3-dev \
    libboost-dev \
    libboost-thread-dev \
    libxmu-dev \
    libxi-dev \
    exuberant-ctags \
    make \
    checkinstall \
    autotools-dev \
    fakeroot \
    xutils \
    cmake \
    autoconf \
    git \
    git-core \
    swig
fi

echo "35" ;

if true ; then
    echo "# Installing various python packages..."
    sudo apt-get -y --force-yes install \
    python2.7 \
    python-tk \
    python-gtk2-dev \
    python-setuptools \
    python-pip \
    python-numpy-dev \
    python-matplotlib \
    python-matplotlib-doc \
    python-scipy \
    libboost-python-dev
fi

echo "40" ;

if true ; then
    echo "# Installing pam keyring packages..."
    sudo apt-get -y --force-yes install \
    libpam0g-dev \
    libpam-dev \
    libpam-gnome-keyring \
    libpam-keyring \
    libpam-modules
fi

# # # # # # # # # # # # Installs 2 # # # # # # # # # # # # # # # # # # #
echo "45" ;

if true ; then
    echo "# Installing extra apt-get keys/repos/programs"
    # keys
    sudo add-apt-repository ppa:nilarimogard/webupd8
    sudo add-apt-repository ppa:app-review-board/ppa
    sudo add-apt-repository ppa:tualatrix/ppa
    sudo add-apt-repository ppa:nilarimogard/webupd8
    sudo add-apt-repository ppa:bisigi
    sudo add-apt-repository ppa:rabbitvcs/ppa
    sudo add-apt-repository ppa:kokoto-java/usu-extras
    sudo add-apt-repository ppa:danielrichter2007/grub-customizer
    sudo add-apt-repository ppa:atareao/atareao
    sudo add-apt-repository ppa:ferramroberto/gimp
    sudo add-apt-repository ppa:kokoto-java/usu-extras
    sudo add-apt-repository ppa:tiheum/equinox
    # update
    sudo apt-get -y --force-yes update
    sudo apt-get -y --force-yes upgrade
    # install
    sudo apt-get -y --force-yes install \
	launchpad-getkeys \
	ubuntu-tweak \
	gimp-plugin-registry \
	eco-theme \
        tropical-theme \
        rabbitvcs-core \
        rabbitvcs-nautilus \
        rabbitvcs-cli \
        mechanig \
        grub-customizer \
        touchpad-indicator \
        gimp-gmic \
        mechanig \
        faenza-icon-theme
fi

echo "50" ;

# not fully automatic yet, need to configure manually
if false ; then
    # dropbox
    if ! test -d "$HOME/Dropbox" ; then
        echo "# Dropbox installation... "
        wget http://www.dropbox.com/download?dl=packages/nautilus-dropbox_0.6.7_amd64.deb
        sudo dpkg --install nautilus-dropbox_*.deb
        killall nautilus
        sleep 4
        dropbox start -i
    fi
fi

# # # # # # # # # # # # Tweaks # # # # # # # # # # # # # # # # # # # # #
echo "55" ;

if true ; then
    # speed up gnome menu
    echo "# GTK setting to speed up the menus... "
    file="$HOME/.gtkrc-2.0"
    if ! grep "gtk-menu-popup-delay" "$file" >/dev/null ; then
        echo "gtk-menu-popup-delay = 0" > ~/.gtkrc-2.0
    fi
fi

echo "60" ;

if true ; then
    # Linux process scheduler
    #   NOW ONLY WORKS IN GRUB 2 ! ! !
    if (! grep "elevator=deadline" /boot/grub/grub.cfg > /dev/null 2>&1); then
        echo "# Changing kernel process scheduler to deadline"
        sudo sed -ie '/_DEFAULT/s/splash/splash elevator=deadline/' /etc/default/grub
        sudo update-grub
    fi
fi

echo "65" ;

if true ; then
    # parallel booting
    echo "# parallel booting"
    sudo perl -i -pe 's/CONCURRENCY=none/CONCURRENCY=shell/' /etc/init.d/rc
fi

echo "70" ;

if false ; then
    # window close button placement
    echo "# move window buttons to right"
btnxml='<?xml version="1.0"?>
<gconf>
<entry name="button_layout" mtime="1281817076" type="string">
<stringvalue>:minimize,maximize,close</stringvalue>
</entry>
</gconf>'
    echo "$btnxml" > ~/.gconf/apps/metacity/general/%gconf.xml
fi

# # # # # # # # # # # # system config # # # # # # # # # # # # # # # # # # # # #

echo "80" ;

# not fully automatic yet, need to select manually
if false ; then
    # enhanced shell, better for bash-isms in makefiles
    echo "# enchancing bash to dash"
    sudo dpkg-reconfigure --force dash
fi

echo "85" ;

# may need work!
if true ; then
    # remap capslock to control
    xmodmap -e 'clear Lock'
    xmodmap -e 'keycode 0x66 = Control_L' # modify 0x42 (find via xev)
    xmodmap -e 'add Control = Control_L'
fi

echo "90" ;

if false ; then
    # remove old nvidia drivers
    echo "# removing default nvidia packages"
    # apt
    sudo apt-get -y --force-yes purge nvidia-*
    # remove non-cuda nvidia drivers
    if (! grep "nvidiafb" /etc/modprobe.d/blacklist.conf > /dev/null 2>&1); then
        sudo echo "               "  >> /etc/modprobe.d/blacklist.conf
        sudo echo blacklist vga16fb  >> /etc/modprobe.d/blacklist.conf
        sudo echo blacklist nouveau  >> /etc/modprobe.d/blacklist.conf
        sudo echo blacklist rivafb   >> /etc/modprobe.d/blacklist.conf
        sudo echo blacklist nvidiafb >> /etc/modprobe.d/blacklist.conf
        sudo echo blacklist rivatv   >> /etc/modprobe.d/blacklist.conf
    fi
    echo "# configuring cuda library"
    # cuda install
    if (! grep "cuda" /etc/modprobe.d/blacklist.conf > /dev/null 2>&1); then
        #sudo echo /usr/local/cuda/lib >> /etc/ld.so.conf
        sudo echo /usr/local/cuda/lib64 >> /etc/ld.so.conf
    fi
    # ldconfig
    sudo ldconfig
fi

# # # # # # # # # # # # Cleanup # # # # # # # # # # # # # # # # # # # # #
echo "95" ;

if true ; then
    # Clean up after the installs.
    echo "# Cleaning packages... "
    sudo apt-get -y --force-yes clean
    sudo apt-get -y --force-yes autoclean
    sudo apt-get -y --force-yes autoremove
fi

# temp dir
$echo_exec rm -rf "$dir"

# # # # # # # # # # # # Finish # # # # # # # # # # # # # # # # # # # # #

# done
echo "100" ;
echo " ===== "
echo " ===== Finished. Please restart."
echo " ===== "

