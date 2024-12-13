on run {tabName, imagePath}
    tell application "System Events"
        set isRunning to (count of (every process whose name is "Google Chrome")) > 0
    end tell

    if not isRunning then
        tell application "Google Chrome"
            activate
            delay 2
        end tell
    end if

    tell application "Google Chrome"
        set tabExists to false
        repeat with w in every window
            repeat with t in every tab of w
                if (title of t contains tabName) then
                    set tabExists to true
                    tell t to reload
                    exit repeat
                end if
            end repeat
            if tabExists then exit repeat
        end repeat

        if not tabExists then
            tell window 0 to make new tab at the end of tabs with properties {URL:imagePath}
        end if
    end tell
end run
