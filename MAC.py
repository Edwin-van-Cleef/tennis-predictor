import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# LOAD AND INVESTIGATE DATA
# ----------------------------------------------------------------
PlayerStats = pd.read_csv('tennis_stats.csv')

# Add Win/Loss ratio
PlayerStats['Win-Loss Ratio'] = PlayerStats.apply(lambda r: float(r.Wins) / r.Losses if r.Losses > 0 else 0, axis=1)
# View available stats
print('Available Stats:\n')
for stat in PlayerStats.columns.values:
    print(stat)

# Group assumes values are either related to player ID, Variables,or performance indicators
player_data = ['Player', 'Year']
game_data = ['FirstServe', 'FirstServePointsWon',
             'FirstServeReturnPointsWon', 'SecondServePointsWon',
             'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted',
             'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
             'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon',
             'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon',
             'TotalServicePointsWon', 'Win-Loss Ratio']
results = ['Wins', 'Winnings', 'Ranking']

# ________________________________________________________________
# EXPLORATORY ANALYSIS
# ----------------------------------------------------------------
# #UN-COMMENT TO PLOT GRAPHS FOR ALL RELATIONSHIPS
for var in game_data:


    plt.subplots(2, 2, figsize=(4, 4))
    # # RANKING
    ax = plt.subplot(2, 2, 1)
    ax.scatter(PlayerStats[var], PlayerStats.Ranking, alpha=0.5)
    ax.set_xlabel(var)
    ax.set_ylabel("Ranking")
    ax.set_title("Player Ranking against {}".format(var))
    # # WINNINGS
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(PlayerStats[var], PlayerStats.Winnings, alpha=0.5)
    ax2.set_xlabel(var)
    ax2.set_ylabel("Winnings")
    ax2.set_title("Player Winnings against {}".format(var))
    # # WIN-LOSS RATIO
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(PlayerStats[var], PlayerStats['Win-Loss Ratio'], alpha=0.5)
    ax3.set_xlabel(var)
    ax3.set_ylabel('Win-Loss Ratio')
    ax3.set_title("Player Win-Loss Ratio against {}".format(var))
    # # WINS
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(PlayerStats[var], PlayerStats.Wins, alpha=0.5)
    ax4.set_xlabel(var)
    ax4.set_ylabel('Wins')
    ax4.set_title("Player Wins against {}".format(var))

    plt.show()


# These game-data variables show a relationship between
# -Aces
# -Break Points Faced
# -Break Point Opportunities
# -Double Faults
# -Return Games Played
# -Return Games Played
# -Service Games Played



# ________________________________________________________________
# SINGLE LINEAR REGRESSION
# ----------------------------------------------------------------
# select features and value to predict
features = PlayerStats[['BreakPointsOpportunities']]
winnings = PlayerStats[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with BreakPointsOpportunities Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()



# ________________________________________________________________
# DOUBLE LINEAR REGRESSION
# ----------------------------------------------------------------


# select features and value to predict
features = PlayerStats[['BreakPointsOpportunities','FirstServeReturnPointsWon']]
winnings = PlayerStats[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with 2 Features Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 2 Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()




# ________________________________________________________________
# MULTIPLE LINEAR REGRESSION
# ----------------------------------------------------------------

# select features and value to predict
features = PlayerStats[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon',
                        'SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced',
                        'BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed',
                        'ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon',
                        'TotalPointsWon','TotalServicePointsWon']]
winnings = PlayerStats[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with Multiple Features Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Multiple Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

# Set the threshold for 'Good' Correlation:
Threshold = 0.3

# lists of 'Good' Correlations for predicting performance indicators:
# N.B in theory (and by experiment) performance indictors should correlate with each other so I've manually added them in
New_vars_wins = ['Winnings','Ranking']
New_vars_winnings = ['Wins','Ranking']
New_vars_ranking = ['Wins','Winnings']

# iterate through vars to see which are good predictors of each PI:
variables = game_data
indicators = results
for var in variables:
    for Indicator in indicators:
        x = PlayerStats[[var]]
        y = PlayerStats[Indicator]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

        # build regression model
        mlr = LinearRegression()
        mlr.fit(x_train,y_train)
        y_predict = mlr.predict(x_test)

        # plt.scatter(y_test,y_predict,alpha = 0.5)
        # plt.show()
        print('\nMODEL PERFORMANCE FOR: {I} vs. {V}'.format(I = Indicator, V = var))
        print("Train score: {}". format(mlr.score(x_train, y_train)))
        print("Test score: {}". format(mlr.score(x_test, y_test)))

        if mlr.score(x_test, y_test) > Threshold:
            if Indicator == 'Wins':
                New_vars_wins.append(var)
            if Indicator == 'Winnings':
                New_vars_winnings.append(var)
            if Indicator == 'Ranking':
                New_vars_ranking.append(var)
        # Give ranking a reduced threshold
        if mlr.score(x_test, y_test) > Threshold/3 and Indicator == 'Ranking':
            New_vars_ranking.append(var)


print('\nGOOD PREDICTORS FOR WINS ARE:')
for i in New_vars_wins:
    print('-{}'.format(i))
print('\nGOOD PREDICTORS FOR WINNINGS ARE:')
for i in New_vars_winnings:
    print('-{}'.format(i))
print('\nGOOD PREDICTORS FOR RANKINGS ARE:')
for i in New_vars_ranking:
    print('-{}'.format(i))


# __________________________________THOUGHTS____________________________________
# Wins are the easiest to predict, with winnings not that much harder
# Rankings are incredibly hard to predict with a lower threshold require to yield any 'good correlators'


# ________________________________________________________________
# MULTIPLE LINEAR REGRESSION
# ----------------------------------------------------------------

# use our good correlators to predict PIs:
for Indicator, Vars in zip(indicators, [New_vars_wins, New_vars_winnings, New_vars_ranking]):
    x = PlayerStats[Vars]
    y = PlayerStats[Indicator]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

    # build regression model
    mlr = LinearRegression()
    mlr.fit(x_train,y_train)
    y_predict = mlr.predict(x_test)

    # Plot Datapoints
   
    ax = plt.subplot()
    plt.scatter(y_test,y_predict,alpha = 0.5)
    ax.set_xlabel('{} Test Values'.format(Indicator))
    ax.set_ylabel('{} Predicted Values'.format(Indicator))
    ax.set_title("Optimised Test vs. Predicted Values for {I} in Tennis\nTest score: {S}".format(I = Indicator,S = mlr.score(x_test, y_test)))
    plt.savefig("Optimised Test vs. Predicted Values for {} in Tennis.png".format(Indicator))
    plt.show()
    print('\nOPTIMISED MODEL PERFORMACE FOR: {I}'.format(I = Indicator))
    print("Train score: {}". format(mlr.score(x_train, y_train)))
    print("Test score: {}". format(mlr.score(x_test, y_test)))



# __________________________________THOUGHTS____________________________________
# -Improvements for Wins and Winnings
# -Rankings still very hard to predict from the current data
# -the Model Predicts negative rankings, which is impossible
# -Other Performance indicators were not considered as variables, but in theory they should correlate with each other.
# -Adding them in improves the test score by about 3-4%






















