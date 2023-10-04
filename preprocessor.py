import csv

# Function to pre-process the data file and return a list of the matches
def preprocess_dataset(filename, output_filename, n_matches=1000):

    with open(filename, 'r') as f:
        content = f.read()

    data = []
    for line in content.split('\n'):
        if line.startswith('    <TwoPlayerGame'):

            date_time = line.split('EndTime="')[1].split('"')[0]

            player1 = line.split('Player1="')[1].split('"')[0]

            # Check if line has a Player1Score
            if 'Player1Score' in line:
                player1_score = int(line.split('Player1Score="')[1].split('"')[0])
            else:
                player1_score = 0
            
            player2 = line.split('Player2="')[1].split('"')[0]

            # Check if line has a Player2Score
            if 'Player2Score' in line:
                player2_score = int(line.split('Player2Score="')[1].split('"')[0])
            else:
                player2_score = 0

            data.append((date_time, player1, player1_score, player2, player2_score))

    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Time', 'Player 1', 'Player 2', 'Player 1 Score', 'Player 2 Score'])
        for i, match in enumerate(data):
            date, time = match[0].split(' ')
            writer.writerow([date, time, match[1], match[3], match[2], match[4]])

            if i == n_matches - 1:
                break
