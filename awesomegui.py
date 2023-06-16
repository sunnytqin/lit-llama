from tkinter import *
from tkinter import ttk
import numpy as np
import argparse

idx = 0
top_list = []
link_list = []
prompt_list = []
window=Tk()

def main(args):
    file_path = args.data
    data = np.load(file_path)
    prompts = data['prompts']

    JSD = data['JSD']
    large_sentence_pieces = data['large_sentence_pieces']
    small_sentence_pieces = data['small_sentence_pieces']
    large_sentence_probs = data['large_sentence_probs']
    small_sentence_probs = data['small_sentence_probs']

    WIDTH = 1000
    HEIGTH = 500
    N_GEN = len(data['JSD'])

    window.title('Awesome GUI')
    window.geometry(f"{WIDTH}x{HEIGTH}")
    window.configure(bg='gray17')
    
    def close():
        global window
        window.destroy()

    def help():
        global window
        top= Toplevel(window)
        Label(top, text="prompt",font=('Courier', 15), fg="yellow", justify='left', wraplength=0.9*WIDTH).grid(row=1,column=1)
        Label(top, text="LLM generated text",font=('Courier', 15), fg="white", justify='left', wraplength=WIDTH, bd=-2).grid(row=2,column=1)
        Label(top, text="LLM generated text (large difference)",font=('Courier', 15), fg="red", justify='left', wraplength=WIDTH, bd=-2).grid(row=3,column=1)
        Label(top, text="user clicked text",font=('Courier', 15), fg="green", justify='left', wraplength=WIDTH, bd=-2).grid(row=4,column=1)
        Label(top, text="user clicked text",font=('Courier', 15), fg="purple", justify='left', wraplength=WIDTH, bd=-2).grid(row=5,column=1)
        Label(top, text="Click on tokens to see logits and other info. \n use reset to reset color. \n next and prev to see different prompts",font=('Courier', 15), fg="white", justify='left', wraplength=WIDTH, bd=-2).grid(row=6,column=1)

    def reset():
        global top_list, link_list, idx
        if 0 < idx < N_GEN:
            for l in link_list:
                if l['fg'] == 'green':  l['fg'] = 'white'
                if l['fg'] == 'purple':  l['fg'] = 'red'
            for t in top_list: t.destroy()   
        else:
            for l in link_list: l.destroy()  


    help_button = Button(window, text="Help", command=help)
    help_button.pack()
    help_button.place(relx=0, y=10)

    next_button = Button(window, text="<Prev", command=lambda inc=-1: next_page(inc))
    next_button.pack()
    next_button.place(relx=0.3, y=10)

    reset_button = Button(window, text="Reset", command=reset)
    reset_button.pack()
    reset_button.place(relx=0.4, y=10)

    prev_button = Button(window, text="Next>", command=lambda inc=1: next_page(inc))
    prev_button.pack()
    prev_button.place(relx=0.5, y=10)

    close_button = Button(window, text="Close", command=close)
    close_button.pack()
    close_button.place(relx=0.9, y=10)


    def open_popup(idx, i, link, w, h):
        global top_list, window
        top= Toplevel(window)
        top_list.append(top)

        if link['fg'] == 'white': link['fg'] = 'green'
        if link['fg'] == 'red': link['fg'] = 'purple'
        root_x = window.winfo_rootx()
        root_y = window.winfo_rooty()
        top.geometry(f"500x250+{int(root_x+w)}+{int(root_y+h+20)}")
        top.title("Token Detail")
        names = ['small', 'Prob', 'large', 'Prob']
        top.pack_propagate(False)
        for c in range(len(names)):
            Label(top, text=names[c], font=('Courier 15 bold')).grid(row=1,column=c+1)

        for r in range(2, 6):
            j = r - 2
            Label(top, text=f"{((small_sentence_pieces[idx, j, i]).replace('▁', ' '))}", font=('Courier 15')).grid(row=r,column=1)
            Label(top, text=f"{small_sentence_probs[idx, i, j]:.2f}", font=('Courier 15')).grid(row=r,column=2)
            Label(top, text=f"{((large_sentence_pieces[idx, j, i]).replace('▁', ' '))}", font=('Courier 15')).grid(row=r,column=3)
            Label(top, text=f"{large_sentence_probs[idx, i, j]:.2f}", font=('Courier 15')).grid(row=r,column=4)
        
        Label(top, text='JSD', font=('Courier 15 bold')).grid(row=8,column=1)
        Label(top, text='Entropy (Small)', font=('Courier 15 bold')).grid(row=8,column=2)
        Label(top, text='Entropy (Large)', font=('Courier 15 bold')).grid(row=8,column=3)
        Label(top, text=f"{JSD[idx, i]:.2f}", font=('Courier 15')).grid(row=9,column=1)

        Button(top, text="Close", command=lambda top=top, link=link: close_and_reset(top, link)).grid(row=10,column=3)#button to close the window
        
    def close_and_reset(top, link):
        top.destroy()
        if link['fg'] == 'green':  link['fg'] = 'white'
        if link['fg'] == 'purple':  link['fg'] = 'red'

    def next_page(inc):
        #Create a button in the main Window to open the popup
        global link_list, top_list, prompt_list, idx, window
        for l in link_list:
            l.destroy() 
        for t in top_list:
            t.destroy()   
        for p in prompt_list:
            p.destroy()      

        if idx + inc < 0:
            link = Label(window, text="You've reached the beginning",font=('Courier', 15), fg="red", justify='left', wraplength=WIDTH, bd=-1)
            link.pack()
            link.place(x=0, y=50)
            link_list.append(link)
        elif idx + inc >= N_GEN:
            link = Label(window, text="You've reached the end",font=('Courier', 15), fg="red", justify='left', wraplength=WIDTH, bd=-1)
            link.pack()
            link.place(x=0, y=50)
            link_list.append(link)
        else:
            idx = min(max(idx + inc, 0), N_GEN -1)
            cum_length = 0 
            line_index = 50
            top_list = []
            link_list = []
            prompt_list = []

            prompt = prompts[idx]
            prompt = Label(window, text=prompt,font=('Courier', 15), fg="yellow", justify='left', wraplength=0.9*WIDTH)
            prompt.place(x=10,y=line_index)
            prompt_list.append(prompt)
            window.update()

            cum_length += (prompt.winfo_width())%(0.9*WIDTH) 
            line_index += prompt.winfo_height() 

            for i in range(len(large_sentence_pieces[idx, 0, :])):
                if JSD[idx][i]>0.5:
                    link = Label(window, text=(small_sentence_pieces[idx, 0, i]).replace("▁", " "),font=('Courier', 15), fg="red", justify='left', wraplength=WIDTH, bd=-2)
                else:
                    link = Label(window, text=(small_sentence_pieces[idx, 0, i]).replace("▁", " "),font=('Courier', 15), fg="white", justify='left', wraplength=WIDTH, bd=-2)
                link.pack(padx=0, pady=0)
                link.bind("<Button-1>", lambda e, idx=idx, i=i, link=link, w=cum_length, h=line_index:open_popup(idx, i, link, w, h))
                link.place(x=cum_length, y=line_index)
                link_list.append(link)
                window.update()
                if cum_length > 0.9*WIDTH:
                    cum_length = 0
                    line_index += 20
                else:
                    cum_length += link.winfo_width()
    next_page(0)
    window.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Awesome GUI')
    parser.add_argument('--data', dest='data', type=str, help='path to the LLM output file generated by generate.py')

    args = parser.parse_args()
    main(args)
