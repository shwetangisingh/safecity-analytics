# SafeCity MCP Server (Claude Code)

This folder contains the MCP server for crime-category prediction:

- Server entrypoint: `src/mcp/server.py`
- Tool name: `predict_crime_category`

## Quick Setup (Claude Code)

From the project root:

```bash
claude mcp add -s project safecity-crime-predictor -- python3 src/mcp/server.py
claude mcp get safecity-crime-predictor
```

If the second command shows `Status: Connected`, the server is ready.

## Notes

- The server uses `models/random_forest_model.pkl` by default.
- If that model is missing/corrupted, the server rebuilds a fallback random-forest model automatically from `data/processed/crime_data_cleaned.csv`.
- If cleaned data does not exist yet, run:

```bash
python3 src/data_cleaning.py
```

## Quick Health Check

After adding the MCP server, ask Claude Code to call:

- `server_health`
- `list_crime_categories`
- `predict_crime_category` (with any valid test payload)

## Remove Server

```bash
claude mcp remove safecity-crime-predictor -s project
```
