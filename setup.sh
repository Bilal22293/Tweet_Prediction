#!/bin/bash

GOOGLE_DRIVE_LINK="https://drive.google.com/uc?export=download&id=1ccbfbfOmuXnCerw7S6Tz7F_D3W4pUc3B"

download_from_google_drive() {
    local url=$1
    local output=$2

    local gdurl=$(echo $url | sed 's|/file/d/|/uc?export=download&id=|' | sed 's|/view.*||')

    echo "Downloading......."
    curl -c ./cookie -s -L "$gdurl" > /dev/null
    local confirm=$(awk '/download/ {print $NF}' ./cookie)
    curl -L -o "$output" -b ./cookie "$gdurl&confirm=$confirm"
    rm ./cookie
}

ZIP_FILE_NAME="Team59_Adobe.zip"
download_from_google_drive $GOOGLE_DRIVE_LINK $ZIP_FILE_NAME
unzip $ZIP_FILE_NAME -d ./
rm $ZIP_FILE_NAME

echo "Setup completed.Zip file downloaded!! Ready to rock!!"
