#/bin/sh

echo "lawn login..."
USERNAME=user
PASSWORD=pswd
tries=4

for i in `seq 1 $tries`; 
do
	echo "login attempt"
	echo $i
	#STATUS=`curl -s -f -F username="$USERNAME" -F password="$PASSWORD" -F iss='true' -F output='text' https://auth.lawn.gatech.edu/index.php`	
	STATUS=`wget -q -O - --post-data="username=${USERNAME}&password=${PASSWORD}&iss=true&output=cli" https://auth.lawn.gatech.edu/index.php`
	echo $STATUS
	sleep 1
done

echo "done"
exit 0
