"""Summarize the ADARP Empatica E4 sensor dataset."""

from load_data import find_sessions, load_session


def main():
    sessions = list(find_sessions())
    print(f"{len(sessions)} sessions across "
          f"{len({p for p, _, _ in sessions})} participants")

    participant, session_id, path = sessions[2]
    data = load_session(path)
    print(f"\n{participant} / {session_id}")
    for name, frame in data.items():
        if name == "tags":
            print(f"  {name:5s} {len(frame)} event marks")
        else:
            span = f"{frame.index[0]} -> {frame.index[-1]}" if len(frame) else "empty"
            print(f"  {name:5s} {len(frame):>7,} samples  {span}")


if __name__ == "__main__":
    main()
