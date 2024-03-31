# Open the source log file
with open("f278d666-a525-415f-945b-8633f75a604b.log", "r") as source_file:
    # Open (or create) the target txt file
    with open("price_history.txt", "w") as target_file:
        # Variable to track whether the "Activities log:" section has been found
        activities_log_found = False
        
        # Read through each line in the source file
        for line in source_file:
            # Check if we've found the "Activities log:" section
            if "Activities log:" in line:
                activities_log_found = True
                # Skip writing the "Activities log:" line itself
                continue
            
            # If the "Activities log:" section was found, write to the target file
            if activities_log_found:
                if line == '\n':  # If the line is just a newline, we've reached the end of the section
                    break  # Stop reading from the source file
                else:
                    target_file.write(line)  # Write the current line to the target file
