from decimal import Decimal
from http.client import HTTPException
from botocore.exceptions import ClientError

class User:
    def __init__(self, user_id, bankroll, bet_history=None, pitch_index=0, table=None):
        self.user_id = user_id
        self.bankroll = bankroll or 100.0
        self.bet_history = bet_history or []
        self.pitch_index = pitch_index or 0
        self.table = table

    # if new user, create function that will save the user to the database with the default bankroll and empty bet history
    def save_to_db(self):
        item = {
            "user_id": self.user_id,
            "bankroll": Decimal(str(self.bankroll)),
            "pitch_index": Decimal(str(self.pitch_index)),
            "bet_history": self.bet_history
        }

        # create if not exists, prevents races
        self.table.put_item(Item=item,
                       ConditionExpression="attribute_not_exists(user_id)",
        )

    def place_bet(self, bet):
        if bet.amount > self.bankroll:
            raise ValueError("Insufficient funds to place bet")
        self.bankroll -= bet.amount
        self.bet_history.append(bet)
    
    def update_bankroll(self, won, amount):
        if won == True:
            self.bankroll += amount
        
        else:
            self.bankroll -= amount

        # update bankroll in database
        try:
            self.table.update_item(
                Key={"user_id": self.user_id},
                UpdateExpression="SET bankroll = :new_bankroll",
                ExpressionAttributeValues={":new_bankroll": Decimal(str(self.bankroll))}
            )
        except ClientError as e:
            raise HTTPException(500, "Failed to update bankroll in database")
    
    def advance_pitch(self):
        self.pitch_index += 1
        # update the pitch index in the database
        try:
            self.table.update_item(
                Key={"user_id": self.user_id},
                UpdateExpression="SET pitch_index = :new_index",
                ExpressionAttributeValues={":new_index": Decimal(str(self.pitch_index))}
            )
        except ClientError as e:
            raise HTTPException(500, "Failed to advance pitch index in database")

    def get_bankroll(self):
        return self.bankroll
    
    def get_bet_history(self):
        return self.bet_history
    
    def get_pitch_index(self):
        return self.pitch_index
