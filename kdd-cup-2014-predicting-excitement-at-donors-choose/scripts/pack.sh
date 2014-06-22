version=$1 ; [ x$version == x ] && version=current
rm ../answers/$version.txt.zip
zip ../answers/$version.txt.zip ../answers/$version.txt
cp ../answers/$version.txt.zip /var/www/tmp/
