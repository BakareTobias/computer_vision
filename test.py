from pynput.keyboard import Key, Listener

def show(key):
    if key == :
        print('c')
    if key == Key.delete:
        return False

with Listener(on_press=show) as listener:
    listener.join()

