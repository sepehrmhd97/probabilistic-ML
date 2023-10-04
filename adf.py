import csv
from gibbs import *

def run_adf(filename, n_iters, burn_in, mu, sigma, sigma_t):

    # Dictionary to store the skill parameters for each team in the form of a mean and std
    # i.e. teams['team_name'] = [mean, std]
    team_skills = {}

    # Parse the match data from the CSV file and run the Gibbs sampler per match
    # in order to update the prior distribution of the skill parameters
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:

            if (i + 1) % 100 == 0:
                print("Processed {} matches".format(i + 1))

            # Skip the first row
            if i == 0:
                i += 1
                continue

            date, time, team1, team2, score1, score2 = row

            score1 = int(score1)
            score2 = int(score2)

            # Skip the match if the score is a draw
            if (score1 == score2):
                i += 1
                continue

            # Skip the match if the score is a draw
            if (score1 == score2):
                i += 1
                continue

            # Get team1 and team2's skill parameters
            mu1, sigma1 = team_skills.get(team1, [mu, sigma])
            mu2, sigma2 = team_skills.get(team2, [mu, sigma])

            # If team 1 has won
            if score1 > score2:
                y = 1
            # If team 2 has won
            elif score1 < score2:
                y = -1

            # Run the Gibbs sampler
            t1_samples, t2_samples = gibbs_sampler(n_iters, burn_in, mu1, mu2, sigma1, sigma2, sigma_t, y)

            # Update the team's skill parameters
            team_skills[team1] = [np.mean(t1_samples[burn_in:]), np.std(t1_samples[burn_in:])]
            team_skills[team2] = [np.mean(t2_samples[burn_in:]), np.std(t2_samples[burn_in:])]

            i += 1

    return team_skills

# Improved ADF algorithm where a draw is also considered as a possible outcome
def run_adf_improved(filename, n_iters, burn_in, mu, sigma, sigma_t):

    # Dictionary to store the skill parameters for each team in the form of a mean and std
    # i.e. teams['team_name'] = [mean, std]
    team_skills = {}

    # Parse the match data from the CSV file and run the Gibbs sampler per match
    # in order to update the prior distribution of the skill parameters
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:

            if (i + 1) % 100 == 0:
                print("Processed {} matches".format(i + 1))

            # Skip the first row
            if i == 0:
                i += 1
                continue

            date, time, team1, team2, score1, score2 = row
            
            score1 = int(score1)
            score2 = int(score2)

            # Get team1 and team2's skill parameters
            mu1, sigma1 = team_skills.get(team1, [mu, sigma])
            mu2, sigma2 = team_skills.get(team2, [mu, sigma])

            # If team 1 has won
            if score1 > score2:
                y = 1
            # If team 2 has won
            elif score1 < score2:
                y = -1
            # If the match is a draw
            else:
                y = 0

            # Get update coefficients based on the scores of the teams
            team_1_update_coeff = 1 + abs(score1 - score2) / 3
            team_2_update_coeff = 1 + abs(score2 - score1) / 3

            # Run the Gibbs sampler
            t1_samples, t2_samples = gibbs_sampler(n_iters, burn_in, mu1, mu2, sigma1, sigma2, sigma_t, y, team_1_update_coeff, team_2_update_coeff)

            # Update the team's skill parameters
            team_skills[team1] = [np.mean(t1_samples[burn_in:]), np.std(t1_samples[burn_in:])]
            team_skills[team2] = [np.mean(t2_samples[burn_in:]), np.std(t2_samples[burn_in:])]

            i += 1

    return team_skills

# Function to predict the result of a match based on the skill parameters of the teams
# Uses the conservative skill metrics based on the TrueSkill paper to predict the result
def predict_result(mu_1, mu_2, sigma_1, sigma_2):
    cons_skill_1 = mu_1 - 3*sigma_1
    cons_skill_2 = mu_2 - 3*sigma_2
    if cons_skill_1 > cons_skill_2:
        return 1
    else:
        return -1
    
# Improved prediction function where a draw is also predicted in addition to a win or loss
def predict_result_improved(mu_1, mu_2, sigma_1, sigma_2):
    cons_skill_1 = mu_1 - 3*sigma_1
    cons_skill_2 = mu_2 - 3*sigma_2
    # If the difference between the conservative skill parameters is less than 0.5, predict a draw
    if abs(cons_skill_1 - cons_skill_2) < 0.005:
        return 0
    if cons_skill_1 > cons_skill_2:
        return 1
    else:
        return -1

# Function to run one-step ahead predictions on the matches and update the skill parameters as it goes
def run_onestep_preds(filename, n_iters, burn_in, mu, sigma, sigma_t):

    # Dictionary to store the skill parameters for each team in the form of a mean and std
    # i.e. teams['team_name'] = [mean, std]
    team_skills = {}

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        correct_preds = 0
        total_nondraw_matches = 0
        for row in reader:
            
            if (i + 1) % 100 == 0:
                print("Processed {} matches".format(i + 1))

            # Skip the first row
            if i == 0:
                i += 1
                continue

            date, time, team1, team2, score1, score2 = row

            score1 = int(score1)
            score2 = int(score2)

            # Skip the match if the score is a draw
            if (score1 == score2):
                i += 1
                continue

            # Get team1 and team2's skill parameters
            mu1, sigma1 = team_skills.get(team1, [mu, sigma])
            mu2, sigma2 = team_skills.get(team2, [mu, sigma])

            # Predict the result
            pred = predict_result(mu1, mu2, sigma1, sigma2)

            if score1 > score2:
                if pred == 1:
                    correct_preds += 1
                y = 1

                
            elif score1 < score2:
                if pred == -1:
                    correct_preds += 1
                y = -1

            # Run the Gibbs sampler
            t1_samples, t2_samples = gibbs_sampler(n_iters, burn_in, mu1, mu2, sigma1, sigma2, sigma_t, y)

            # Update the team's skill parameters
            team_skills[team1] = [np.mean(t1_samples[burn_in:]), np.std(t1_samples[burn_in:])]
            team_skills[team2] = [np.mean(t2_samples[burn_in:]), np.std(t2_samples[burn_in:])]
            total_nondraw_matches += 1

            i += 1

    print("Accuracy of one-step ahead predictions: {}".format(correct_preds / total_nondraw_matches))

    return team_skills


# Improved one-step ahead prediction function where a draw is also considered
def run_onestep_preds_improved(filename, n_iters, burn_in, mu, sigma, sigma_t):

    # Dictionary to store the skill parameters for each team in the form of a mean and std
    # i.e. teams['team_name'] = [mean, std]
    team_skills = {}

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        correct_preds = 0
        for row in reader:
            
            if (i + 1) % 100 == 0:
                print("Processed {} matches".format(i + 1))

            # Skip the first row
            if i == 0:
                i += 1
                continue

            date, time, team1, team2, score1, score2 = row

            score1 = int(score1)
            score2 = int(score2)

            # Get team1 and team2's skill parameters
            mu1, sigma1 = team_skills.get(team1, [mu, sigma])
            mu2, sigma2 = team_skills.get(team2, [mu, sigma])

            # Predict the result
            pred = predict_result_improved(mu1, mu2, sigma1, sigma2)

            if score1 > score2:
                if pred == 1:
                    correct_preds += 1
                y = 1

                
            elif score1 < score2:
                if pred == -1:
                    correct_preds += 1
                y = -1

            else:
                if pred == 0:
                    correct_preds += 1
                y = 0

            # Get update coefficients based on the scores of the teams
            team_1_update_coeff = abs(score1 - score2) / 3
            team_2_update_coeff = abs(score2 - score1) / 3

            # Run the Gibbs sampler
            t1_samples, t2_samples = gibbs_sampler(n_iters, burn_in, mu1, mu2, sigma1, sigma2, sigma_t, y, team_1_update_coeff, team_2_update_coeff)

            # Update the team's skill parameters
            team_skills[team1] = [np.mean(t1_samples[burn_in:]), np.std(t1_samples[burn_in:])]
            team_skills[team2] = [np.mean(t2_samples[burn_in:]), np.std(t2_samples[burn_in:])]

            i += 1

    print("Accuracy of one-step ahead predictions: {}".format(correct_preds / i))

    return team_skills
            