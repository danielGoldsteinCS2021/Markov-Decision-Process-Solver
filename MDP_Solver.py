def stateValueFinder(matrix, actions, policy, discount, theta):
    rowLen, colLen = len(matrix), len(matrix[0])
    while True:
        newMatrix, delta = [[0 for _ in range(colLen)] for _ in range(rowLen)], 0
        for i in range(rowLen):
            for j in range(colLen):
                newStateValue = 0
                for action in actions:
                    rewardAndLocation = determineRewardAndSPrimeLocation(action, i, j, rowLen, colLen)
                    reward, sPrimeRow, sPrimeCol = rewardAndLocation[0], rewardAndLocation[1][0], rewardAndLocation[1][1]
                    if (i, j) == (0, 1):
                        sPrimeRow, sPrimeCol = 4, 1
                    if (i, j) == (0, 3):
                        sPrimeRow, sPrimeCol = 2, 3
                    newStateValue += policy * 1 * (reward + discount * matrix[sPrimeRow][sPrimeCol])
                newMatrix[i][j] = newStateValue
                delta = max(delta, abs(abs(matrix[i][j]) - abs(newStateValue)))
        matrix = newMatrix
        if delta <= theta:
            print("Returned")
            printMatrix(matrix)
            return matrix
    print("Not Returned")
    printMatrix(matrix)
    return matrix


def determineRewardAndSPrimeLocation(action, row, col, rowLen, colLen):
    reward, location = 0, (row, col)
    if (row, col) == (0, 1):
        reward = 10
    if (row, col) == (0, 3):
        reward = 5
    if action == 'LEFT' and col > 0:
        location = (row, col - 1)
    if action == 'RIGHT' and col + 1 < colLen:
        location = (row, col + 1)
    if action == 'UP' and row > 0:
        location = (row - 1, col)
    if action == 'DOWN' and row + 1 < rowLen:
        location = (row + 1, col)
    if row == location[0] and col == location[1] and reward == 0:
        reward = -1
    return reward, location


def printMatrix(matrixToPrint):
    for row in matrixToPrint:
        for value in row:
            print(round(value, 1), end=' ')
        print()


m = [[0 for j in range(5)] for i in range(5)]
a = ['left'.upper(), 'right'.upper(), 'up'.upper(), 'down'.upper()]
m0 = stateValueFinder(m, a, 0.25, 0.9, 0.000000000000000000001)

