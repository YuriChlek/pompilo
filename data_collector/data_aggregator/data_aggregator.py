from utils import get_db_pool

async def calculate_avg_size():
    conn = await get_db_pool()

    # Запит на обчислення середнього значення size у двох таблицях
    query = """
            SELECT COUNT(*) as total_count, AVG(size) as avg_size
            FROM (
                SELECT size FROM bybit_trading_history_data.solusdt_p_trades
                UNION ALL
                SELECT size FROM binance_trading_history_data.solusdt_p_trades
            ) AS all_sizes;
        """

    result = await conn.fetchrow(query)
    avg_size = result['avg_size']
    total_count = result['total_count']

    print(f"Загальна кількість записів: {total_count}")
    print(f"Середнє значення size з обох таблиць: {avg_size:.4f}")


async def start_data_aggregator():
    await calculate_avg_size()
