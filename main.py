import multiprocessing

def main():
    import flet as ft
    from app import main_gui
    ft.app(target=main_gui)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        print("실행 에러:", e)
        import traceback; traceback.print_exc()