
-- 1. Count total customers
SELECT COUNT(*) AS total_customers
FROM customers;

-- 2. Count total products
SELECT COUNT(*) AS total_products
FROM products;

-- 3. Count orders by status
SELECT status, COUNT(*) AS total_orders
FROM orders
GROUP BY status;

-- 4. Calculate completed order revenue
SELECT ROUND(SUM(order_amount), 2) AS total_revenue
FROM orders
WHERE status = 'completed';

-- 5. Find the top 10 customers by completed-order spending
SELECT
    customer_id,
    COUNT(*) AS total_orders,
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
ORDER BY total_spend DESC
LIMIT 10;

-- 6. Count customers by city
SELECT city, COUNT(*) AS total_customers
FROM customers
GROUP BY city
ORDER BY total_customers DESC;
