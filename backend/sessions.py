import os, time
import boto3
from datetime import datetime, timezone
from decimal import Decimal
from botocore.exceptions import ClientError
from fastapi import HTTPException

DEFAULT_BANKROLL = float(os.getenv("DEFAULT_BANKROLL", "100"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(6 * 60 * 60)))

# create function to return DynamoDB table storing user session information
def get_table():
    dynamodb = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))
    table = dynamodb.Table(os.getenv("SESSIONS_TABLE", "PitchBettingSessions"))
    return table

def create_user(user_id, table):
    item = {
        "user_id": user_id,
        "bankroll": Decimal(str(DEFAULT_BANKROLL)),
        "pitch_index": Decimal("0"),
        "bet_history": []
    }
    # create if not exists (prevents races)
    table.put_item(
        Item=item,
        ConditionExpression="attribute_not_exists(user_id)",
    )
    return item

# create a function that can retrieve users from the DynamoDB table
def get_user(user_id):
    table = get_table()
    resp = table.get_item(Key={"user_id": user_id})
    if "Item" in resp:
        item = resp["Item"]

        # "created_at": now_iso(),
        # "updated_at": now_iso(),
        # "ttl": ttl_epoch(SESSION_TTL_SECONDS),

    else:
        item = create_user(user_id, table)

    return item

def advance_pitch(user_id, i):
    table = get_table()

    try:
        table.update_item(
            Key={"user_id": user_id},
            UpdateExpression="""
                SET pitch_index = pitch_index + :one
            """,
            ConditionExpression="pitch_index = :expected",
            ExpressionAttributeValues={
                ":one": Decimal("1"),
                ":expected":Decimal(str(i))
  
            },
            # ReturnValues="ALL_NEW",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            # Another request already advanced the pitch (double-click, rerun, etc.)
            raise HTTPException(409, "Pitch already advanced; refresh and try again")
        raise

# def calc_bet_profit():

# def update_budget():