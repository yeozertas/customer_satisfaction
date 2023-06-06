# Customer Satisfaction

Extract whl with

```bash
python setup.py bdist_wheel
```

Install as package

```bash
python setup.py install
```

Example usage

```python
from customer_satisfaction import CustomerSatisfaction
cs_m = CustomerSatisfaction()

# Intent results
print(cs_m.get_script_intents())

# Sentiment results
print(cs_m.get_script_sentiments())
```

