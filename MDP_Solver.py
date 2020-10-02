# Daniel Goldstein
# 20119615
# CISC 474 Assignment 1

# stateValueFinder(2dArray, Array, Float, Float, Float, Dictionary) -> Matrix
# stateValueFinder finds the state values for each entry in the matrix passed to it
# it does so based on the actions, policy, discount, theta and specialPositions provided
def stateValueFinder(matrix, actions, policy, discount, theta, specialPositions):
    rowLen, colLen = len(matrix), len(matrix[0])
    while True:
        newMatrix, delta = [[0 for _ in range(colLen)] for _ in range(rowLen)], 0
        for i in range(rowLen):
            for j in range(colLen):
                newStateValue = 0
                for action in actions:
                    rewardAndLocation = determineRewardAndSPrimeLocation(action, i, j, rowLen, colLen, specialPositions)
                    reward, (sPrimeRow, sPrimeCol) = rewardAndLocation[0], rewardAndLocation[1]
                    newStateValue += policy * 1 * (reward + discount * matrix[sPrimeRow][sPrimeCol])
                newMatrix[i][j] = newStateValue
                delta = max(delta, abs(matrix[i][j] - newStateValue))
        matrix = newMatrix
        if delta <= theta:
            break
    return matrix


# determineRewardAndSPrimeLocation(Array, Int, Int, Int, Int, Dictionary) -> Int, (Int, Int)
# returns reward for provided action and the location the agent will be in after taking said action
def determineRewardAndSPrimeLocation(action, row, col, rowLen, colLen, specialPositions):
    reward, location = 0, (row, col)
    if (row, col) == specialPositions['A'][0]:
        reward, location = 10, specialPositions['A'][1]
    elif (row, col) == specialPositions['B'][0]:
        reward, location = 5, specialPositions['B'][1]
    elif action == 'LEFT' and col > 0:
        location = (row, col - 1)
    elif action == 'RIGHT' and col + 1 < colLen:
        location = (row, col + 1)
    elif action == 'UP' and row > 0:
        location = (row - 1, col)
    elif action == 'DOWN' and row + 1 < rowLen:
        location = (row + 1, col)
    elif row == location[0] and col == location[1]:  # we will hit a wall if we move
        reward = -1
    return reward, location


# printMatrix(2dArray, Boolean) -> None
# prints provided matrix and rounds values if roundMatrix == True
def printMatrix(matrixToPrint, roundMatrix=True):
    for row in matrixToPrint:
        for value in row:
            if roundMatrix:
                print(round(value, 1), end=' ')
            else:
                print(value, end=' ')
        print()


# main() -> None
# runs code to solve 5x5, 7x7 MDPs
def main():
    actions = ['left'.upper(), 'right'.upper(), 'up'.upper(), 'down'.upper()]
    policy, theta = 0.25, 0.0000001
    specialPositions5x5 = {'A': [(0, 1), (4, 1)], 'B': [(0, 3), (2, 3)]}
    specialPositions7x7 = {'A': [(2, 1), (6, 1)], 'B': [(0, 5), (3, 5)]}
    m0 = [[0 for _ in range(5)] for _ in range(5)]
    m1 = [[0 for _ in range(7)] for _ in range(7)]
    m5x5_75 = stateValueFinder(m0, actions, policy, 0.75, theta, specialPositions5x5)
    m5x5_85 = stateValueFinder(m0, actions, policy, 0.85, theta, specialPositions5x5)
    m7x7_75 = stateValueFinder(m1, actions, policy, 0.75, theta, specialPositions7x7)
    m7x7_85 = stateValueFinder(m1, actions, policy, 0.85, theta, specialPositions7x7)

    print("State Value For 5x5, Discount Rate 0.75, Rounded To 1 Decimal Place")
    printMatrix(m5x5_75)
    print("\nNot Rounded")
    printMatrix(m5x5_75, roundMatrix=False)
    print("\nState Value For 5x5, Discount Rate 0.85, Rounded To 1 Decimal Place")
    printMatrix(m5x5_85)
    print("\nNot Rounded")
    printMatrix(m5x5_85, roundMatrix=False)
    print("\nState Value For 7x7, Discount Rate 0.75, Rounded To 1 Decimal Place")
    printMatrix(m7x7_75)
    print("\nNot Rounded")
    printMatrix(m7x7_75, roundMatrix=False)
    print("\nState Value For 7x7, Discount Rate 0.85, Rounded To 1 Decimal Place")
    printMatrix(m7x7_85)
    print("\nNot Rounded")
    printMatrix(m7x7_85, roundMatrix=False)


if __name__ == "__main__":
    main()
