import asyncio

async def get_gpt_description(df) -> str:
    # print 100 first characters of string
    print(str(df)[:100])

    await asyncio.sleep(3)

    return "This is GPT-42 recommendation"
