#/bin/sh

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

echo "Recursively removing backup files from:"
pwd
echo "REMOVING:"
sudo find ./ -name '*~' -exec rm '{}' \; -print
sudo find ./ -name '*.tmp' -exec rm '{}' \; -print
sudo find ./ -name '*.bak' -exec rm '{}' \; -print
sudo find ./ -name '*.log' -exec rm '{}' \; -print
echo "FINISHED"

echo "Emptying Trash..."
rm -rf /home/*/.local/share/Trash/*/** &> /dev/null
rm -rf /root/.local/share/Trash/*/** &> /dev/null
echo "Done!"
echo ""


