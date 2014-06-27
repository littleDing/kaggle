FILE=$1

cat $FILE | grep -a -e "aucs" | sed 's/.*\(aucs=.*\)/\1/g' | head
head $FILE | grep -a -e "INFO {" | sed 's/.*INFO \(.*\)/\1/g'
cat $FILE | grep -a -e "train_x=" | tail -n1 | sed 's/.*INFO \(.*\)/\1/g'
cat $FILE | grep -a -e "INFO . fold begins" -e "INFO . fold finished" | awk '
FNR==1{ b=$1" "$2; gsub("-"," ",$1); gsub(":"," ",$2); bt=mktime($1" "$2)}
FNR==2{ e=$1" "$2; gsub("-"," ",$1); gsub(":"," ",$2); et=mktime($1" "$2)}
	END{
	print b,"=>",e,(et-bt)/60
	}'
