#!/bin/bash

# marc brumlik, tailored software inc, Mon Sep  8 22:50:20 CDT 2008
# version 0.48 Wed Jun 10 23:36:35 CDT 2009
# tsi-inc@comcast.net

# convert various image, audio, and video files into other likely formats

# set -x

#####
# Watcher
#####
avwatcher() {
### this function produces "Preparing" progress window
### during conversion, this window appears until the output file exists

# exit 0

sleep 1
pid=`ps -ef | grep " $4 " | grep "$2" | awk '{print $2}'`
z="0"
while true
do
	if test -s "$1"
		then	break
	fi
        for a in 10 20 30 40 50 60 70 80 90 80 70 60 50 40 30 20 10 00
        do
                test -s "$1" && break 2
                echo $a; sleep 1
		z=`expr $z + 1`
		if [ "$z" = "20" ]
			then	exit 0
		fi
        done
done | zenity --progress --auto-close --text="Waiting for $2\nto begin writing $1" || ( kill -9 $pid >/dev/null 2>&1; zenity --info --text="Avconvert was CANCELLED!" )

while true
do
        newsize=`du -a -b "$1" | awk '{print $1}'`
        progress=`echo "scale=10; $newsize / $3 * 100" | bc`
	if [ "$progress" = "$oldprogress" ]
		then	break
	fi
	oldprogress=$progress
	sleep 1
	echo $progress
done | zenity --progress --percentage=01 --auto-close --text="Writing to $1\nEstimate to completion..." || ( kill -9 $pid >/dev/null 2>&1; zenity --info --text="Avconvert was CANCELLED!" )
}
# end of watcher

#####
# Convert
#####
myavconvert() {
#
# processing images or performing text-to-image conversion

# set -x

# this is for looping after the first item, using the env for settings
looping=n
if test -s /tmp/avconvert.env
	then	. /tmp/avconvert.env
		looping=y
fi

# the output types allowed for

case $it in
	image)  out=" gif jpg ico pdf png tif OTHER"
		height=460 ;;
	text)   out=" gif jpg ico pdf png tif OTHER"
		height=370 ;;
esac

# set default checkbox: all false except the original ext (except txt)
case $imageext in
	none)	out=`echo "$out" | sed -e 's/ / FALSE /g'` ;;
	same)	out=`echo "$out" | sed -e 's/ / FALSE /g' -e "s/FALSE \$ext/TRUE $ext/"` ;;
	*)	out=`echo "$out" | sed -e 's/ / FALSE /g' -e "s/FALSE \$imageext/TRUE $imageext/"` ;;
esac

# show output choices to user and loop until selection is made
while true
do
if [ "$looping" = "n" ]
	then
case $prog$it in
        convertimage)   ident=`identify "$target"`
			choice=`zenity --list --height=$height --title="Converting file $target" --text="Source file format $show,\n$ident\n\nYou can select multiple formats - avconvert will loop to create them all.\n\n\OUTPUT NAMING: names of the files created will include the proper extension,\nand if the resolution differs from the original then a \"-XXX\" before the extension.\n\nChoose OTHER for an output type not listed" --checklist --column "Select" --column "Convert to format" $out` || exit 0 ;;
        converttext)    choice=`zenity --list --height=$height --title="Convert $target" --text="Source file format $show\n\nDefault action is to convert TEXT to IMAGE\n\nChoose OTHER for an output type not listed" --checklist --column "Select" --column "Convert to format" $out` || exit 0 ;;
esac
fi

if [ -n "$choice" ]
	then	break
fi
done

# was it OTHER?
case $choice in
        OTHER_non-image)  exec myavtext ;;
        OTHER)  choice=`zenity --entry --title="Output type" --text="Supply an appropriate file extension"` || exit 0 ;;
esac

# clean up any _audio reference and any leading . in ext
choice=`echo $choice | sed -e 's/_.*//' -e 's/^\..*//'`

#       loop over (possibly multiple) choice(s)
# (THE OUTER LOOP)
for each in `echo $choice | sed 's/|/ /g'`
do

#       what will the destination filename be?
new=`echo $target | sed -e "s/.$ext$//" -e "s/$/.$each/"`

if [ "$looping" = "n" ]
	then
# set resolution here
# results into $dim, will look like "jpg|gif|pdf"
dim=`zenity --list --height=540 --title="Resolution" --text="Output resolution for \"$new\"\n(Resolution of $target is\n$ident)\n\nThis will be the dimension along the longer edge\n(depending on portrait or landscape mode)\n\nChoose one or more, or leave all blank to leave resolution unchanged" --checklist --column "Select" --column "Choose Resolution" FALSE 1280 FALSE 1024 FALSE 800 FALSE 640 FALSE 480 FALSE 320 FALSE 250 FALSE 200 FALSE 150 FALSE 100 FALSE "Something else"` || exit 0
fi

case "$dim" in
	Something*)	while true
			do
				dim=`zenity --entry --text="New size (along longer edge)"` || exit 0
				case $dim in
					[0-9]*[0-9])	break ;;
					*)		;;
				esac
			done ;;
esac

# if $safedims is set and $dim is empty
# then set it to what we used last time
if [ -z "$dim" -a -n "$safedims" ]
	then	dims="$safedims"
fi
# parse $dim into $dims
case $dim in
	'')	dims=" " ;;
	*)	dims=`echo $dim | sed 's/|/ /g'` ;;
esac

# and save these $dims for next time
safedims="$dims"
textdims="Previously selected resolutions: $safedims"

# if this is a jpeg, ask for quality
if [ "$looping" = "n" ]
	then
case "$new" in
	*jpg)	qual=`zenity --list --height=290 --title="JPEG Quality" --text="Choose JPEG quality for \"$new\"" --radiolist --column "Select" --column "Choose Quality" FALSE 100 FALSE 90 FALSE 80 TRUE 70 FALSE 60 FALSE 50` || exit 0 ;;
esac
fi

# create arg to convert for quality
case "$new" in
	*jpg)	case $qual in
			'')	quality="-quality 70" ;;
			*)	quality="-quality $qual" ;;
		esac ;;
esac

# loop through all the $dims.
for rez in " "$dims
do
# THE INNER LOOP

case "$rez" in
	" ")	geom="" ;;
	*)	rez=`echo $rez`; geom="-geometry $rez"x"$rez"
		new=`echo $target | sed -e "s/.$ext$//" -e "s/$/.$rez.$each/"` ;;
esac
if test -s "$outdir/$new"
	then	 zenity --question --text "$new exists - Overwrite?" || new=`zenity --file-selection --save --confirm-overwrite --title="Choose a new file/folder"` || exit 0
fi

case $it in
text)	# EXTRA stuff for TEXT to IMAGE
	colorlist=`convert -list color | sed -n '/,/s/  .*//p' | sort -u`
	colorlist=`echo " "$colorlist | sed 's/ / FALSE /g'`
	if whereis kcolorchooser | grep / >/dev/null
		then	colorlist="FALSE GUI-\"kcolorchooser\" $colorlist"
		else	colorlist="--- Install_\"kdegraphics\"_for_GUI $colorlist"
	fi
	colorlist="FALSE Key-in_RGB-value $colorlist"
	# for OLD version of convert
	# fontlist=`convert -list font | sed -n '/0$/s/ .*//p' | sort -u`
	# for NEW version of convert
	fontlist=`convert -list font | grep Font | sed 's/^.*://' | \
		sed -e 's/^ //g' -e 's/ $//' | sort -u`
	fontlist=`echo " "$fontlist | sed 's/ / FALSE /g'`
	sizelist=" 8 10 12 15 20 25 30 35 40 50 60 80 100 120 150 200 250 300"
	sizelist=`echo "$sizelist" | sed 's/ / FALSE /g'`

	back=`zenity --list --height=500 --title="Background" --text="Choose Background Color\n\nNOTE: If you use GUI chooser, the RGB values\nwill be shown in subsequent windows\nfor your reference.\n\nALSO NOTE: for some unknown reason\nthere is a long delay when GUI closes." --radiolist --column "Select" --column "Choose Color" $colorlist` || exit 0
	case $back in
		GUI-*)	back=`kcolorchooser --print` ;;
		_GUI*)	zenity --info --title="kcolorchooser" --text="To choose colors with a GUI\ninterface, install the \"kdegraphics\" package"; exit 0 ;;
		Key-*)	back=`zenity --entry --title="Key-in Color" --text="Enter the RGB color as:  #RGB  #RRGGBB  #RRRGGGBBB  #RRRRGGGGBBBB  rgb(rrr,ggg,bbb)"` || exit 0 ;;
	esac
	fill=`zenity --list --height=500 --title="Foreground" --text="[Background is $back]\n\nChoose Foreground Color" --radiolist --column "Select" --column "Choose Color" $colorlist` || exit 0
	case $fill in
		GUI-*)	fill=`kcolorchooser --print` ;;
		_GUI*)	zenity --info --title="kcolorchooser" --text="To choose colors with a GUI\ninterface, install the \"kdegraphics\" package"; exit 0 ;;
		Key-*)	fill=`zenity --entry --title="Key-in Color" --text="Enter the RGB color as:  #RGB  #RRGGBB  #RRRGGGBBB  #RRRRGGGGBBBB  rgb(rrr,ggg,bbb)"` || exit 0 ;;
	esac
	borw=`zenity --list --height=340 --title="Border Width" --text="[Background is $back]\n[Foregeround color is $fill]\n\nChoose Border Width" --radiolist --column "Select" --column "Choose Width" FALSE 0 FALSE 5 FALSE 10 FALSE 20 FALSE 30 FALSE 50` || exit 0
	case $borw in
		0)	borc="transparent" ;;
		*)	borc=`zenity --list --height=500 --title="Border Color" --text="[Background is $back]\n[Foreground color is $fill]\n[Border width is $borw]\n\nChoose Border Color" --radiolist --column "Select" --column "Choose Color" $colorlist` || exit 0 ;;
	esac
	case $borc in
		GUI-*)	borc=`kcolorchooser --print` ;;
		_GUI*)	zenity --info --title="kcolorchooser" --text="To choose colors with a GUI\ninterface, install the \"kdegraphics\" package"; exit 0 ;;
		Key-*)	borc=`zenity --entry --title="Key-in Color" --text="Enter the RGB color as:  #RGB  #RRGGBB  #RRRGGGBBB  #RRRRGGGGBBBB  rgb(rrr,ggg,bbb)"` || exit 0 ;;
	esac
	font=`zenity --list --height=500 --title="Font" --text="[Background is $back]\n[Foregeround color is $fill]\n[Border width is $borw]\n[Border color is $borc]\n\nChoose Text Font and Style" --radiolist --column "Select" --column "Choose Font" $fontlist` || exit 0
	size=`zenity --list --height=500 --title="Font Size" --text="[Background is $back]\n[Foregeround color is $fill]\n[Border width is $borw]\n[Border color is $borc]\n[Font style is $font]\n\nChoose Text Font Size" --radiolist --column "Select" --column "Choose Size" $sizelist` || exit 0
	textoptions="-background $back -fill $fill -border $borw -bordercolor $borc -font $font -pointsize $size label:@"
	longestline=`cat "$target" | awk 'BEGIN {max[i]=0}
                        (length >= max[i]) {max[i] = length}
                        END {printf "%s\n", max[i]}'`
	# horizrez=`echo -e "scale=0\n$rez * 8 / 11\nquit\n" | bc`
	horizrez=$rez
	maxpossible=`echo -e "scale=0\n$horizrez / $size\nquit\n" | bc`
	# splitpieces=`echo -e "scale=0\n$longestline / $maxpossible\nquit\n" | bc`
	newlength=`echo -e "scale=2\n$maxpossible * .9\nquit\n" | bc`
	newlength=`echo "$newlength" | sed 's/\..*//'`
	if [ "$longestline" -ge "$maxpossible" ]
		then	newtarget=`echo "$target" | sed "s/$ext$/wrap$newlength/"`
			zenity --info --title="Adjusting input file" --text="The source file \"$target\" contains long lines.\n\nThe longest line is $longestline but the max possible\nwith a width of $horizrez and a size of $size is $maxpossible.\n\nCreating an intermediate text file with a word wrap at\nthe first space after character position $newlength.\n\nThis will get you APPROXIMATELY the\ncharacter size you asked for.\nThe intermediate file will contain the full text of $target.\n\nIntermediate file name is: $newtarget"
			sed "s/.\{$newlength\} /&\n/g" < "$target" > "$newtarget"
			target="$newtarget"
	fi

	linecount=`wc $target | awk '{print $1}'`
	maxpossible=`echo -e "scale=2\n$rez / $size\nquit\n" | bc`
	maxpossible=`echo -e "scale=2\n$maxpossible * 2 / 3\nquit\n" | bc`
	maxpossible=`echo "$maxpossible" | sed 's/\..*//'`

	if [ "$linecount" -gt "$maxpossible" ]
		then	shorttarget=`echo $target | sed 's/$ext$//'`
			zenity --info --title="Adjusting input file" --text="The source file \"$target.txt\" contains too many lines.\nAdjusting for a more normal aspect ratio.\n\nThere are $linecount lines, but the max possible\nwith a length of $rez and a size of $size is $maxpossible.\n\nSplitting $target.txt into a set of files with $maxpossible\nlines in each.  The resulting files will be processed.\n\nThis will get you APPROXIMATELY the\ncharacter size you asked for.\n\nIntermediate files will be named as $shorttarget"aa.txt", $shorttarget"ab.txt", $shorttarget"ac.txt", etc."
			cat "$target" | split -$maxpossible - "$shorttarget"
			targets=`ls $shorttarget??`
			multiple=y
			for each in `ls $shorttarget??`
			do
				mv "$each" "$each.txt"
			done
	fi
	;;
esac
#End of extra stuff for text to image

# Now, do the actual conversion

# usually just one target
targets="$target"

# in a text->image that needed splitting will be multiple $targets
case $multiple in
	y)	count=1
		targets=`ls $shorttarget??*`
		basenew="$new" ;;
esac

echo "$targets" | \
while read target
do
case $multiple in
	y)	case $count in
			?)	count="00"$count ;;
			??)	count="0"$count ;;
		esac
		new=`echo "$basenew" | sed "s/\....$/.$count&/"`
		;;
esac
oldsize=`du -a -b "$target" | awk '{print $1}'`
avwatcher "$outdir/$new" "$prog" "$oldsize" $$ &
convert -verbose $geom $quality $textoptions"$target" "$outdir/$new" 2>/tmp/convert.$$.$count | zenity --progress --text="Working on $new" --auto-close --auto-kill --pulsate
# ImagMagick/convert may call ufraw for camera raw images
# there is a new version of ufraw with slightly different options
# convert is using old options, causing a "deprecated" complaint
# but the results are good as old options are still accepted.
cleanup=`egrep -v "deprecated|ppm16" /tmp/convert.$$.$count` >/tmp/$$
mv /tmp/$$ /tmp/convert.$$.$count
if test -s /tmp/convert.$$.$count
	then	message=`cat /tmp/convert.$$.$count`
		zenity --info --text="An error occured during the conversion\n\nThe message was: $message" &
	else	rm -f /tmp/convert.$$.$count
fi

case $multiple in
	y)	count=`expr $count + 1` ;;
esac
# end of multiple $targets loop
done

# end of inside loop for $rez
done
# end of outside loop for $dims
done 

# convert -verbose $geom $quality $textoptions"$target" "$outdir/$new" 2>/tmp/convert.$$.$count | zenity --progress --text="Working on $new" --auto-close --auto-kill --pulsate
case $autoloop in
	y)	echo "	choice=\"$choice\"
			dim=\"$dim\"
			dims=\"$dims\"
			geom=\"$geom\"
			qual=\"$qual\"
			textoptions=\"$textoptions\"" > /tmp/avconvert.env ;;
esac
}
# end of avconvert



#####
# ffmpeg
#####
myavffmpeg() {
#
# processing audio and video

# set -x
# this is for looping after the first item, using the env for settings
looping=n
if test -s /tmp/avffmpeg.env
	then	. /tmp/avffmpeg.env
		looping=y
fi

# set up output extensions
case $av in
	video)  out=" asf avi flv m4p mov mpg wmv m4a_audio_only mp3_audio_only wav_audio_only wma_audio_only jpg_frames png_frames OTHER"
		height=600 ;;
	audio)  out=" m4a mp3 wav wma OTHER"
		height=360 ;;
esac

out=`echo "$out" | sed -e 's/ $ext //' -e "s/ / FALSE /g"`

# loop for output conversion type
if [ "$looping" = "n" ]
	then
while true
do
choice=`zenity --list --height=$height --title="Convert $target" --text="Source file format $show\n$ffident\n\nChoose OTHER for an output type not listed\n" --radiolist --column "Select" --column "Convert to format" $out` || exit 0

if [ -n "$choice" ]
        then      break
fi
done
fi

# was it OTHER?
case $choice in
        OTHER)  choice=`zenity --entry --title="Output type" --text="Supply an appropriate file extension"` || exit 0 ;;
esac


case $choice in
	*_audio*)	av=audio; choice=`echo $choice | sed 's/_.*//'` ;;
	*_frames)	av=frame; choice=`echo $choice | sed 's/_.*//'`
			outf=`echo $choice | awk -F_ '{print $1}'`
			step=`zenity --scale --text="Seconds between capture\n\"0\" means all frames" --value=1 --min-value=0 --max-value=60 --step=1` || exit 0
			;;
esac

# have the output type

# audio bitrate
if [ "$looping" = "n" ]
	then
bits=""
case $av in
        audio)  bits=`zenity --list --height=270 --title="Select" --text="Change bitrate? (select nothing to leave as-is):" --radiolist --column "Select" --column "Bitrate" FALSE 32k FALSE 64k FALSE 96k FALSE 128k FALSE 192k FALSE 256k` || exit 0 ;;
esac
case $bits in
        '')     ;;
        *)      bits="-ab $bits" ;;
esac
fi

new=`echo $target | sed "s/$ext$/$dot$choice/"`

if test -s "$outdir/$new"
	then     zenity --question --text "$outdir/$new exists - Overwrite?" || new=`zenity --file-selection --save --confirm-overwrite --title="Choose a new file/folder"` || exit 0
fi

case $av in
	frame)	prog=mplayer; new=00000001.$outf ;;
esac
oldsize=`du -a -b "$target" | awk '{print $1}'`
avwatcher "$outdir/$new" "$prog" "$oldsize" $$ &
case $av in
	frame)	outf=`echo $outf | sed 's/jpg/jpeg/'`
		mplayer -sstep $step -ao null -vo $outf "$target" 2>&1 | tr '\015' '\012' | tee /tmp/mplayer.$$ | zenity --progress --text="Working on $new" --auto-close --auto-kill --pulsate
		result=`grep A: /tmp/$prog.$$ | awk '{print $2}'`
		result="Captured frames at the following seconds:\n"`echo $result`
		;;
	*)	ffmpeg -y $bits -i "$target" "$outdir/$new" 2>&1 | tr '=' '\012' | tee /tmp/ffmpeg.$$ | zenity --progress --text="Working on $new" --auto-close --auto-kill --pulsate
		result=`tail -5 /tmp/$prog.$$`
		;;
esac

newsize=`du -a -b "$new" | sed 's/   .*//'`
# show results summary
case $quiet in
	y)	notify="--notification --window-icon=/usr/share/zenity/zenity-progress.png"
		cp /usr/share/zenity/zenity.glade.progress-under /usr/share/zenity/zenity.glade 2>/dev/null ;;
	n)	notify="--info"
		cp /usr/share/zenity/zenity.glade.progress-over /usr/share/zenity/zenity.glade 2>/dev/null ;;
esac
zenity $notify --title="Results" --text="Completed $target --> $new
File size $newsize

$prog last message was:

$result" &

case $autoloop in
	y)	echo -e "choice=\"$choice\"\nbits=\"$bits\"" > /tmp/avffmpeg.env ;;
esac
}
# end of ffmpeg


#####
# ISO conversions
#####
myaviso() {

case $it in
	bchunk)	
		whereis bchunk | grep / >/dev/null || exec zenity --warning --text="Must install \"bchunk\" package first!"
		tracks=`grep TRACK $base.cue | awk '{print $2}'`
		zenity --question --text="BIN/CUE to ISO conversion\nCUE shows tracks $tracks\nProgress will be idicated for first track only\nwhich will be named "$base"01.iso" || exit 0
		new="$outdir/$base"
		if test -s "$new"01.iso
			then    zenity --question --text="$new"01.iso" exists - Overwrite?" || new=`zenity --file-selection --save --confirm-overwrite --title="Choose a new file/folder"` || exit 0
		fi
		prog=bchunk
		oldsize=`du -a -b "$base.bin" | awk '{print $1}'`
		avwatcher "$new"01.iso "$prog" "$oldsize" $$ &
		bchunk "$base.bin" "$base.cue" "$new" >/tmp/bchunk.$$ 2>&1
		;;
        daa*|nrg*|b?i*|cdi*|mdf*|pdi*|ccd*|uif*)
		whereis $it | grep / >/dev/null || exec zenity --warning --text="No program \"$it\" found for processing \"$target\"!\n\nFor ISO conversions, install one or more of these packages...\n\nAcetoneISO for:  b5i cdi mdf nrg pdi\nnrg2iso for:     nrg\ndaa2iso for:     daa\nbchunk for:      bin/cue\nccd2iso for:     cdd\nuif2iso for:      uif\n\ncdd2iso is available at: sourceforge.net/projects/ccd2iso\nuif2iso is available at: aluigi.altervista.org/mytoolz.htm\n\nthe rest are in the repositories."
		base=`echo $target | sed 's/....$//'`
                zenity --question --text="Source file is $target\n$ext to ISO conversion\nwhich will be named $base.iso" || exit 0
                new="$outdir/$base".iso
                if test -s "$new".iso
                        then     zenity --question --text "$new".iso" exists - Overwrite?" || new=`zenity --file-selection --save --confirm-overwrite --title="Choose a new file/folder"` || exit 0
				new="$new".iso
                fi
		prog=$ext"2iso"
		oldsize=`du -a -b "$target" | awk '{print $1}'`
		avwatcher "$new" "$prog" "$oldsize" $$ &
		case $ext in
			b?i|pdi)	new="" ;;
		esac
		$prog "$target" "$new" >/tmp/$prog.$$ 2>&1
		;;
esac

}
# end of iso conversions
#####
# Text conversions
#####
myavtext() {
#
# processing text between formats and into other things

# set -x

case $it in
	text)	# image->avconvert sound->espeak html->txt2tags
		out=" image sound html"; height=300 ;;
	doc)	# text->antiword
		out=" text"; height=300 ;;
	rtf)	# text->unrtf html->unrtf
		out=" text html"; height=300 ;;
	odt)	# text->odt2txt xml->odt2txt
		out=" text xml"; height=300 ;;
esac

out=`echo "$out" | sed -e 's/ $ext //' -e "s/ / FALSE /g"`

# loop for output conversion type
while true
do
choice=`zenity --list --height=$height --title="Convert $target" --text="Source file format $show\n$ffident\n" --radiolist --column "Select" --column "Convert to format" $out` || exit 0

if [ -n "$choice" ]
        then      break
fi
done

# pre-check for installed application
case $it$choice in
	textsound)	if ! whereis espeak | grep / >/dev/null
			then	zenity --warning --title HELP --text "\"espeak\" package needs to be installed"
				exit 0
			fi ;;
	texthtml)	if ! whereis txt2tags | grep / >/dev/null
			then	zenity --warning --title HELP --text "\"txt2tags\" package needs to be installed"
				exit 0
			fi ;;
	doctext)	if ! whereis antiword | grep / >/dev/null
			then	zenity --warning --title HELP --text "\"antiword\" package needs to be installed"
				exit 0
			fi ;;
esac

case $it$choice in
	textimage)	prog=convert; myavconvert 2>/tmp/avconvert.debug ;;
	textsound)	lang=`zenity --list --height=500 --title="Language" --text="Choose language for pronunciation" --radiolist --column "Select" --column "Language" FALSE af_afrikaans FALSE bs_bosnian FALSE cs_czech FALSE cy_welsh FALSE de_german FALSE el_greek TRUE en_english FALSE es_spanish FALSE fi_finnish FALSE fr_french FALSE hi_hindi FALSE hr_croatian FALSE hu_hungarian FALSE is_icelandic FALSE it_italian FALSE la_latin FALSE mk_macedonian FALSE nl_dutch FALSE no_norwegian FALSE pl_polish FALSE pt_brazil FALSE ro_romanian FALSE ru_russian FALSE sk_slovak FALSE sr_serbian FALSE sv_swedish FALSE sw_swahihi FALSE vi_vietnam FALSE zh_Mandarin` || exit 0
			lang=`echo $lang | sed 's/_.*//'`
			voice=`zenity --list --height=400 --title="Voice" --text="Choose voice to use" --radiolist --column "Select" --column "Voice" FALSE m1_male-1 FALSE m2_male-2 FALSE m3_male-3 FALSE m4_male-4 FALSE f1_female-1 TRUE f2_female-2 FALSE f3_female-3 FALSE f4_female-4 FALSE croak FALSE whisper` || exit 0
			voice=`echo $voice | sed 's/_.*//'`
			pitch=`zenity --scale --text="Pitch" --min-value=0 --max-value=99 --value=50` || exit 0
			speed=`zenity --scale --text="Speed" --min-value=0 --max-value=99 --value=43` || exit 0
			speed=`expr $speed \* 4`
			new=`echo "$target" | \
				sed -e 's/.txt$//' -e 's/$/.wav/'`
			oldsize=`du -a -b "$target" | awk '{print $1}'`
			avwatcher "$outdir/$new" "$prog" "$oldsize" $$ &
			cat "$target" | espeak --stdin -v$lang+$voice -p $pitch -s $speed -w "$outdir/$new" >/tmp/espeak.$$ 2>&1
			;;
	texthtml)	new=`echo "$target" | \
				sed -e 's/.txt$//' -e 's/$/.html/'`
			oldsize=`du -a -b "$target" | awk '{print $1}'`
			avwatcher "$outdir/$new" "$prog" "$oldsize" $$ &
			txt2tags --infile="$target" --outfile="$outdir/$new" -t xhtml >> /tmp/txt2tags.$$ 2>&1
			if test -s /tmp/txt2tags.$$
			then	zenity --info --title="txt2tags output" --text=`cat /tmp/txt2tags.$$`
			fi
			rm -f /tmp/txt2tags.$$
			;;
	doctext)	new=`echo "$target" | \
				sed -e 's/.doc$//' -e 's/$/.txt/'`
			incl=`zenity --list --height=300 --title="Options" --text="Include either of these?\n\nr = \"revisioning text\"\ns = \"hidden text\"" --checklist --column "Select" --column "Text type" FALSE r FALSE s ` || exit 0
			case $incl in
				'')	;;
				*)	opts=`echo "$incl" | sed 's/|/ /g' | \
					sed 's/^/ /' | sed 's/ / -/g'` ;;
			esac
			oldsize=`du -a -b "$target" | awk '{print $1}'`
			avwatcher "$outdir/$new" "$prog" "$oldsize" $$ &
			antiword $opts "$target" > "$outdir/$new" 2>>/tmp/antiword.$$
			if test -s /tmp/antiword.$$
			then	zenity --info --title="antiword output" --text=`cat /tmp/antiword.$$`
			fi
			rm -f /tmp/antiword.$$
			;;
	*)	zenity --info --title="OOPS!" --text="Not ready yet"; exit 0 ;;

esac
}
# end of text



#####
# Initial script
#####
PATH=$PATH:.; export PATH

# check some stuff first...
# if zenity is not installed, use xterm to display a warning
whereis zenity | grep / >/dev/null || exec xterm -hold -geometry 50x2 -bg red -T "OOPS! Must install zenity first!" -e true
# if imagemagick is not installed, display warning
whereis convert | grep / >/dev/null || exec zenity --warning --text="Must install \"imagemagick\" package first!"
# if ffmpeg is not installed, display warning
whereis ffmpeg | grep / >/dev/null || exec zenity --warning --text="Must install \"ffmpeg\" package first!"
# if mplayer is not installed, display warning
whereis mplayer | grep / >/dev/null || exec zenity --warning --text="Must install \"mplayer\" package first!"

if test -s ~/.avconvert
	then	: defaults already set
	else	echo -e "destdir=n\nautoloop=n\nzenityfix=n\nquiet=n" > /tmp/avc
		behavior=`zenity --list --height=320 --title="Set defaults" --text="This is your first run of avconvert.\n\nThe file ~/.avconvert can be set some default behaviors.  You can create one now by selecting from the options below,\nand you can change it at any time by editing it directly or by removing it, which will trigger this dialogue to reappear.\n\nRead each option below and choose your prefrence followed by OK, or click Cancel to defer this until later." --checklist --column="Select" --column="Description" TRUE "Always pop up to ask for a destination into which to save converted files (destdir=y)" TRUE "When multiple files are selected, process each one in the same way (autoloop=y)" TRUE "Check zenity defaults and offer to fix them so popups are OVER rather than UNDER (zenityfix=y)" TRUE "Quiet mode - use 'notification' instead of 'info' to show some conversion results (quiet=y)"` || exit 0
echo "$behavior" | tr '|' '\012' | sed -e 's/^.*(//' -e 's/.$//' >> /tmp/avc
		imageext=none
		imageext=`zenity --list --height=370 --title="Set defaults" --text="When converting images, the system can set a default for the type of file to be converted to.\n\nchoose one of the following" --radiolist --column="Select" --column="Default image extension" TRUE "No default should be pre-selected (imageext=none)" FALSE "Create an image of the same type as the source (imageext=same)" FALSE "Default to creating gif (imageext=gif)" FALSE "Default to creating jpg (imageext=jpg)" FALSE "Default to creating ico (imageext=ico)" FALSE "Default to creating pdf (imageext=pdf)" FALSE "Default to creating png (imageext=png)" FALSE "Default to creating tif (imageext=tif)"` || exit 0

echo "$imageext" | tr '|' '\012' | sed -e 's/^.*(//' -e 's/.$//' >> /tmp/avc
. /tmp/avc
rm -f /tmp/avc

echo "# Defaults file for avconvert, created `date`
destdir=$destdir		# Always ask for a destination directory
autoloop=$autoloop		# Always treat multiple files the same way
zenityfix=$zenityfix		# Always offer to fix zenity pop-under
imageext=$imageext		# Always default to this extension for images
quiet=$quiet			# Make progress and results less intrusive
" > ~/.avconvert
fi

destdir=y; autoloop=x; zenityfix=y; imageext=none; quiet=n

if test -s ~/.avconvert
	then	. ~/.avconvert
fi

while true
do
	case $1 in
		--no-destdir)	destdir=n; shift ;;
		--no-loop)	autoloop=n; shift ;;
		--no-zenityfix)	zenityfix=n; shift ;;
		*)		break ;;
	esac
done

case $1 in
	'')	files=`zenity --title="Select Source File(s).  They should all be of the same type (eg, all images even if a mix of gif/jpg/png)" --file-selection --multiple --separator="|"` || exit 0
		files=`echo "$files" | sed -e 's/^/"/' -e 's/$/"/' -e 's/|/" "/g'`
		echo "avconvert $* $files" > /tmp/avconvert.rerun
		sh /tmp/avconvert.rerun &
		rm -f /tmp/avconvert.rerun
		exit 0
		;;
esac

case $zenityfix in
	n)	;;
	*)
if test -s /usr/share/zenity/zenity.glade.orig
	then	: already done
	else	cat << ZenityFix > /tmp/zenity.fix
DIR=/usr/share/zenity
if test -s \$DIR/zenity.glade
	then	cd \$DIR
	else	echo -e "Please modify this script with the correct location of the file 'zenity.glade' and re-execute it.  Press Return \c"; read fred; exit 0
fi
echo -e "\nModifying the file \$DIR/zenity.glade\n\nA safe copy of your file is being saved as zenity.glade.orig"
if test -s zenity.glade.orig
	then	: already there
	else	cp zenity.glade zenity.glade.orig
fi
cat zenity.glade.orig | sed '/focus_on_map/s/False/True/' > zenity.glade
cp zenity.glade zenity.glade.progress-over
progline=\`grep -n "zenity_progress_dialog" zenity.glade | sed 's/:.*//'\`
focusline=\`sed -n "\$progline,\$"p < zenity.glade | grep -n focus_on_map | \
	head -1 | sed 's/:.*//'\`
targetline=\`echo -e "\$progline + \$focusline - 1 \n quit" | bc\`
sed "\$targetline s/True/False/" < zenity.glade.progress-over > zenity.glade.progress-under
chmod 666 /usr/share/zenity/zenity.glade
echo -e "Modification complete.\n\nPress Return \c"; read fred
ZenityFix
		chmod 755 /tmp/zenity.fix
		if whereis xterm | grep / >/dev/null
			then	message="Click OK if you would like this to be run for you right now, or if you prefer, click Cancel to prevent avconvert from doing this.  If you Cancel, avconvert will create a script /tmp/zenity.fix which you can run as \'root\' manually.  This will modify your Zenity defauts so its windows appear on top of any others."
			else	message="This fix can not be auto-run for you because you do not have \'xterm\' installed.  Click Cancel and execute /tmp/zenity.fix as \'root\'. This will modify Zenity defaults so its windows appear on top of any others."
		fi
		zenity --question --title "Zenity Defaults" --text="The Linux utility that avconvert uses to pop up dialogs is called Zenity.  In some releases, these dialogs pop up underneath existing windows.  This makes reading the output and responding to prompts from avconvert very inconvenient.\n\nYour installed Zenity version is one which behaves this way.  For your convenience, a small program named \"zenity.fix\" has just been written in your /tmp directory.\n\n$message\n\nNOTE: this window and some others can be permanently silenced either with command line options or by creating a ~/.avconvert config file.  Read the end of the avconvert script for details.\nFurther, this window will not reappear once the zenity defaults are changed OR as long as the /tmp/zenity.fix file exists." && xterm -e ' echo -e "If for some reason this script fails or the password given is incorrect,\nyou will be prompted to run it again the next time you start avconvert\n\nRoot \c"; su -c /tmp/zenity.fix '
fi
	;;
esac

case $quiet in
	y)	cp /usr/share/zenity/zenity.glade.progress-under /usr/share/zenity/zenity.glade 2>/dev/null ;;
	n)	cp /usr/share/zenity/zenity.glade.progress-over /usr/share/zenity/zenity.glade 2>/dev/null ;;
esac

rm -f /tmp/avconvert.env /tmp/avffmpeg.env

until [ -z "$1" ]
do

targetpath=$1; shift
target=`basename "$targetpath"`
dir=`echo "$targetpath" | sed "s/\/\$target//"`

test -d "$dir" && cd "$dir"
if test -f "$target"
	then	if test -s "$target"
			then	: OK
			else	exec zenity --warning --text="Source file is EMPTY!"
		fi
	else	exec zenity --warning --text="Source file NOT FOUND!"
fi

case $destdir in
	n)	outdir=`pwd` ;;
esac

if test -z "$outdir"
	then	cwd=`pwd`
		outdir=`zenity --file-selection --directory --title "Destination Directory for Converted Files (choose CANCEL to default to source directory)"  --filename=$cwd`
		if test -z "$outdir"
			then	outdir=$cwd
		fi
fi

case "$target" in
	*.*)	;;
	*)	zenity --question --title="What to do?" --text="It would be nice if \"$target\" had an extension...\nRenaming this file is recommended!\n\nOK will guess the file type and proceed.\nCancel will exit so you can rename." || exit 0 ;;
esac

type=`file "$target" | tr '[A-Z]' '[a-z]' | sed 's/^.*://'`
mtype=`file -i "$target" | tr '[A-Z]' '[a-z]' | sed 's/^.*://'`
show=`file "$target" | awk -F: '{print $2}'`
case "$target" in
	*.???|*.????)	ext=`echo "$target" | sed 's/^.*\.//'` ;;
	*)		ext="" ;;
esac
# imagemagick identify does not properly execute mencoder on *avi files
# and instead generated a png image for every frame. 
# until fixed, this like should be commented out.
## ident=`identify "$target" 2>/dev/null | sed "s/^.*$ext //"`
# ffmpeg provides lots of info anyway
ffident=`ffmpeg -i "$target" 2>&1 | tail -4 | head -3`
if echo "$ffident" | grep Unable >/dev/null
	then	ffident=""
fi

#	based on source file, what might the destinations be?
# 	what application will we use to create them?
#	the case wildcards are things we know how to convert FROM
#	"prog" is the actual conversion program
#	"out" is a list of formats we can choose to convert TO
it=""; av=""	# for "image/text" and "audio/video"
# strings from file, and extensions, which imply input file type
case "$type$mtype$ext" in
	*bin)
		base=`echo "$target" | sed 's/.bin$//'`
		if test -s "$base".cue
			then	prog=iso; it=bchunk
		fi ;;
	*cue)
		base=`echo "$target" | sed 's/.cue$//'`
		if test -s "$base".bin
			then	prog=iso; it=bchunk
		fi ;;
	*daa)
		prog=iso; it=daa2iso ;;
	*nrg)
		prog=iso; it=nrg2iso ;;
	*b?i)
		prog=iso; it=b5i2iso ;;
	*cdi)
		prog=iso; it=csi2osi ;;
	*mdf)
		prog=iso; it=mdf2iso ;;
	*pdi)
		prog=iso; it=pdi2iso ;;
	*ccd)
		prog=iso; it=ccd2iso ;;
	*uif)
		prog=iso; it=uif2iso ;;
	*text/plain*|*ASCII*)
		# recognized plain text formats, out to other text or image
		prog="text"; it="text" ;;
	*image*|*pdf|*ico)
		# recognized image formats, output will also be an image
		prog="convert"; it="image" ;;
	*video*|*wmv|*asf|*m4p|*ogg)
		# recognized multimedia formats, output to multimedia or audio
		prog="ffmpeg"; av="video" ;;
	*audio*|*mp3|*wav|*m4a|*wma)
		# recognized audio-only formats, output to audio
		prog="ffmpeg"; av="audio" ;;
	*msword*)
		# micro$oft office word document, v2 / v6 or newer
		prog=text; it="doc" ;;
	text/html)
		prog=text; it="html" ;;
	*text/rtf*)
		prog=text; it="rtf" ;;
	*Unicode*)
		prog=text; it="uni" ;;
	*opendocument*|*openoffice*)
		prog=text; it="odt" ;;
	*openoffice*)
		prog=text; it="oo" ;;
	*)	prog=`zenity --list --title="Unknown!" --text="Not sure what to do with this file!\n\nTarget file: \"$target\"\n\nInformation found:\n$type\n$mtype\n$ffident$ident\n\n\nBut, it is likely that though this file type was not\nexplicitly recognized by avconvert, it can be handled\nby convert (Images) or ffmpeg (Audio/Video).\n\nIf you would like to TRY, choose the utility you feel\nwould be appropriate and avconvert will do its best.\n\n" --radiolist --column "Select" --column "Utility to use" FALSE convert FALSE ffmpeg` || exit 0
		case $prog in
			ffmpeg)	av=`zenity --list --title="Select" --text="Tell ffmpeg whether this file is:" --radiolist --column "Select" --column "Content type" FALSE audio FALSE video` || exit 0 ;;
		esac
		;;
		
esac

# export everything we know so far and launch the appropriate function quality

case $prog in
	ffmpeg|convert)	audiovideo=y ;;
	*)		audiovideo=n ;;
esac
case "$1$audiovideo$autoloop" in
	?*yx)	autoloop=n
		zenity --question --title "Auto-Loop though Source Files??" --text "You have selected multiple source files.  Avconvert will loop through them and convert them all.\n\n*IF* you know that all source files are of the same type\n   (images, audio, etc)\nand *IF* you want them all of them converted the same way\n   (images->1024/jpg, audio->128bit/mp3, etc)\n*THEN* click OK\n\n     But...\n\n*IF* they the source files are of different types\n   (not ALL images or not ALL videos)\nor *IF* you want per-file control of how each source is handled\n*THEN* click Cancel\n\nOK will auto-loop, Cancel will treat each file separately." && autoloop=y
		;;
esac

export dir target ext type mtype show ident ffident it av prog autoloop base
myav$prog

# this 'done' is for looping through multiple selected input files
done

exit 0

#####
# READ ME
#####
avconvert

marc brumlik, tailored software inc, Thu Nov 13 18:50:35 CST 2008
tsi-inc@comcast.net

#######
# IF YOU READ NO FURTHER THAN THIS, please note that when using
# "kcolorchooser" in text-to-image, there is a delay after choosing your
# color before the next window from avconvert comes up.  I have no idea why
# kcolorchooser is behaving this way.
#######

This project started out because I needed to create jpg files of numerous
resolutions from a number of source files and wanted an automatic way to
generate them more quickly.  Then my daughter asked for a way to convert
"unplugged" YouTube files into formats her friends could view.
Combining these two short scripts is where avconvert began.

Upon completion of each use of avconvert, a file for debugging purposes
is written to, with lots of output showing exactly what the script 
has done.  This can help a lot if you are getting unexpected results.
The file name is derived from the conversion, followed by "dot" PID.  In the 
case of text->image and if intermediary files were generated, there will be
another "dot" and a sequence number.  These files will be named:
	/tmp/convert.$$
	/tmp/convert.$$.$count
	/tmp/ffmpeg.$$
	/tmp/avconvert.debug
	/tmp/txt2tags.$$
	/tmp/antiword.$$
and more may be added in the future.

You can create a startup file ~/.avconvert to contain preferences.
Currently there are three items that can be specified there.
A ~/.avconvert which specifies the defaults for these would contain this:
	destdir=y
	autoloop=y
	zenityfix=y
The line "destdir" specifies whether or not there should be a popup prompting
for a destination directory for converted files.  The line "autoloop" determines
whether, if multiple files are selected, you will be prompted and asked if you
want to treat them all the same way.  Y means prompt, N means always treat the
files individually.  THe line "zenityfix" has to do with the prompt which offers
to fix zenity popups so they appear above (instead of below) other items on the
desktop.  Y means detect "popunder" and offer to fix it, N means ignore current
zentiy configuration.

Things you might want to customize (starting from the top of the script):

Too much verbage that gives information about source files?
	comment out the "ident=" and "ffident=" lines

Too much window activity and want to eliminate "progress" window?
	uncomment the "exit 0" at the top of "myavconvert-watch"

To add more extensions to recognized file types for image/video/audio:
	the script uses two different forms of output from the
	"file" command plus the source file extension, to decide
	what type of file it has to work with.  "file" isn not 
	always useful, which is why extensions are needed at all.
	find the comment "# strings from file".  Below that are 
	four cases.  Each contains first something that may have
	been output by "file", followed by a set of extensions.
	Just follow the example to add more extensions.

To add more extensions to the default list of output file types:
	Find the comment line beginning "Show appropriate output".
	Below that are four entries - two for "convert" and two 
	for "ffmpeg".  These are each divided into two more.
	Just add any extension you like into that space-separated
	list (do NOT remove the leading space in the list).

To prevent a "default" output type for images:
	Currently the default output for images is the same as 
	the source file type.  To prevent this, find the comment
	"This line sets default" and, on the line for "convert",
	and comment out the "out=" line below that.

To change the list of image resolutions presented:
	Find the comment "# set resolution here".  Add or 
	remove any resolutions from the list.  Also, setting
	one or more from FALSE to TRUE will cause them to be 
	checkboxed when the window appears.

To change the list of jpg quality settings:
	Find the comment "# if this is a jpeg".  Add or remove.
	You may also set the default by changing a FALSE to TRUE

To change the list of font sizes:
	Modify the line beginning "sizelist=".

To change bitrate choices for ffmpeg audio conversion:
	Find the comment line "# audio bitrate".
	To set a default bitrate, change one of the FALSE to TRUE.

To prevent the final window showing summary after completion:
	(Not recommeded, since this is ALSO where you might see 
	diagnostic output after a failure)
	Find the comment " Show results summary" and comment out the 
	"zenity" line that follows it.


COMMENTS WELCOME!
