
import customtkinter as ctk

class ResultsPage(ctk.CTkFrame):
    def __init__(self, master, log_sink):
        super().__init__(master, corner_radius=0)
        self.grid_rowconfigure(1, weight=1); self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Results / Log", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=18, pady=(18,4), sticky="w")
        self.log_box = ctk.CTkTextbox(self, wrap="word"); self.log_box.grid(row=1, column=0, sticky="nsew", padx=18, pady=12)
        self._sink = log_sink

        btns = ctk.CTkFrame(self); btns.grid(row=2, column=0, sticky="ew", padx=18, pady=(0,12))
        ctk.CTkButton(btns, text="Copy Log", command=self.copy_log, width=110).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Clear Log", command=self.clear_log, width=110).pack(side="left", padx=6)

    def append(self, msg):
        self.log_box.insert("end", msg + "\n"); self.log_box.see("end")

    def copy_log(self):
        text = self.log_box.get("1.0", "end"); self.clipboard_clear(); self.clipboard_append(text)

    def clear_log(self):
        self.log_box.delete("1.0", "end")
