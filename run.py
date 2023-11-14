if __name__ == "__main__":
    import uvicorn
    from config import args

    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)
