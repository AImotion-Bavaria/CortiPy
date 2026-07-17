## 1. Session-Konfiguration: schrittweises Formular

**Problem:** Die Konfigurationsseite fragte alles auf einmal ab – auch Felder, die die
gewählte Hardware gar nicht kennt. Das Formular wirkte „schon ausgefüllt“, weil Streamlit-
Auswahlfelder standardmäßig auf ihre erste Option springen.

**Lösung:** Die Konfiguration wird nun **Schritt für Schritt** aufgedeckt. Jeder Schritt
öffnet erst, wenn der vorherige tatsächlich erfüllt ist (die Freigabe wird aus den Daten
gelesen, nicht über einen „Weiter“-Knopf – so kann das Formular nie hängenbleiben):

1. **Gerät** – bestimmt, was der Rest des Formulars überhaupt bedeutet
2. **Verbindung** – gerätespezifische Einstellungen; UNICORN braucht einen Port
   (bei aktivem „Simulate run“ gelockert, da dann keine Hardware angesprochen wird)
3. **Methode** – was gemessen wird
4. **Aufnahme** – Abtastrate + eine Aufnahmedauer größer als null
5. **Kanäle** – die Methodenparameter, freigeschaltet ab `NumberEEGChannels > 0`
6. **Sitzungsdetails** – Dateiname, Proband, Umgebung + Teilnehmerformular (blockiert nie)

Felder noch nicht sichtbarer Schritte behalten ihre gespeicherten Standardwerte, es geht
also nichts verloren. Gerät und Methode starten bewusst „ungesetzt“, damit es wirklich
etwas aufzudecken gibt. *(Commits `6f72f16`, `a0581b4`)*

---

## 2. Fortschrittsanzeige (Setup-Balken)

**Problem:** Der Balken führte eine eigene Vier-Punkte-Checkliste und hakte „Sampling rate“
und „Electrodes“ schon auf einem leeren Formular ab – beide waren durch die Schema-
Standardwerte „per Konstruktion“ wahr. Es hieß „Setup 2/4“, bevor überhaupt ein Gerät
gewählt war.

**Lösung:** Der Balken zeigt jetzt die **sechs echten Konfigurationsschritte** und liest
sie aus derselben Quelle wie das Formular (`session.config_progress_steps`) – beide können
also nicht mehr widersprechen. Leeres Formular = 0/6; vollständiges = 6/6 mit „bereit“.
Die doppelte Mini-Schrittanzeige im Aufklapp-Bereich wurde entfernt (eine ehrliche Anzeige
ist besser als zwei). Auch die Zusammenfassungs-Karte zeigt „N/A“ statt eines
vorgegaukelten „fs: 100 Hz“, solange kein Gerät gewählt ist. *(Commits `ca6a71c`, `a0581b4`)*

---

## 3. Elektroden-Seite

**Problem (Sichtbarkeit):** Nur vier der acht Spalten passten ins Bild; „Model“ war zu breit
und teilte sich die Zeile mit der Kopf-Karte (Scalp Map). Die **Impedanz-Spalte** schien
„nicht mehr gefüllt“ zu werden – sie war schlicht nie sichtbar.

**Lösung:** Die Tabelle steht jetzt über die **volle Breite** (Scalp Map darunter),
Spaltenbreiten sind in Pixeln gesetzt (Model 340 px, kurze Spalten geben Platz ab),
PosX/PosY verstecken sich hinter „Edit coordinates“. *(Commits `3f882db`, `ca6a71c`,
`465b365`)*

**Problem (falsche Kanäle):** Die Seite listete *alle* Kanäle des Geräts (z. B. 32 bei
ActiCHamp) mit einer „Use channel“-Checkbox. Diese Checkbox tat aber nichts: Der Verstärker
liefert immer seine **ersten N Kanäle** – Kanal 30 ist ohne 1…29 nicht lesbar. Ein
beliebiges Häkchen wurde bei jedem Rerun still zurückgesetzt.

**Lösung:** `NumberEEGChannels` ist die **einzige Wahrheit**. Die Tabelle zeigt genau die
aufgenommenen Kanäle (GND/Ref + Kanal 1…N), die „Use channel“-Spalte ist weg, und die
Kopfzeile sagt, was passiert („Recording the first 3 of 32 ActiCHamp channels“) samt Feld
„Channels to record“. Änderungen werden in die volle Montage eingemischt, sodass höhere
Kanäle Position/Impedanz/Modell behalten. *(Commit `9ededb0`)*

**Problem (kein Gerät gewählt):** Ohne Gerät erschien eine sinnlose „Electrodes ()“-Tabelle
mit Ch1…Ch8. **Lösung:** Es wird nun um die Gerätewahl gebeten, da die Montage aus dem
Gerät entsteht. *(Commit `465b365`)*

---

## 4. Elektroden-Bibliothek & Modellauswahl

**Problem:** Die „Model“-Spalte war gar nicht auswählbar (`disabled=True`) und wurde
automatisch aus der Kategorie abgeleitet. „Mehr Modelle zur Auswahl“ lief also ins Leere.

**Lösung:**
- Model ist jetzt ein echtes **Dropdown**; wählt man ein Modell aus einer anderen Kategorie,
  wandert die Kategorie mit (statt die Wahl zu überschreiben).
- Ein der Bibliothek **unbekanntes Modell** wird behalten, nicht überschrieben – importierte
  Datensätze und ältere Exporte tragen Namen, die dieser Build nie gesehen hat; ein
  Zurücksetzen würde Metadaten verfälschen.
- Bibliothek erweitert: **34 → 66 Modelle, 5 → 6 Kategorien** (u. a. neue Kategorie
  „Subdermal Needle“; g.tec g.SAHARA, actiCAP, BioSemi, Quik-Cap u. v. m.). Jede Kategorie
  endet mit „Other / not listed“. *(Commit `465b365`)*

---

## 5. Geräteeinstellungen (ActiCHamp, UNICORN)

- **ActiCHamp 250 Hz entfernt** – auf dieser Einheit nicht nutzbar; Standard ist jetzt 500 Hz.
- **ActiCHamp-Impedanz blieb 0** – `read_impedances` setzte nie das `measureImpedance`-Flag
  des Producers, der Verstärker ging also nie in den Impedanzmodus. Wird jetzt für die
  Messung gesetzt und danach zurückgenommen.
- **Referenzkanal-Auswahl** entspricht jetzt der **Anzahl EEG-Kanäle** (beide Selektoren).
- **UNICORN** hat eine **feste Hardware-Referenz**: Die Referenz-Selektoren sind gesperrt
  und es wird keine Software-Re-Referenzierung geschrieben – bereits referenzierte Daten
  werden nicht erneut referenziert.
- **Kompaktere Geräteanzeige:** Die drei Felder saßen auf einem 2-Spalten-Raster und
  füllten den halben Panel-Platz nicht; Höhe ~300 px → **148 px**, alles in einer Zeile,
  „Rescan“ neben dem Port. *(Commits `72bc2a2`, `f0c0344`)*

---

## 6. Live-Ansicht während der Messung

**Problem:** Das Pop-out-Fenster der Live-Vorschau **aktualisierte sich nie** – der
matplotlib-Writer hatte keine Refresh-Unterstützung, der plotly-Pfad setzte sie auf
`not interactive` (also genau dann `False`, wenn benötigt). Die „Live“-Vorschau war ein
totes Standbild. Außerdem ignorierte die Vorschau „Simulate run“ und versuchte, den echten
seriellen Port zu öffnen.

**Lösung:**
- Fenster aktualisieren sich automatisch; der Live-Plot rendert **inline** in der Seite und
  hängt nicht mehr am Öffnen eines Pop-outs.
- **Eine Spur pro Kanal** mit Name und Min/Max links (Unicorn-Recorder-Stil), festes
  **10-s-Rollfenster** (ein während der Messung wanderndes Fenster macht Läufe
  unvergleichbar) und ein **Y-Achsen-Dropdown** (Auto + feste ±µV-Stufen).
- Ein **„Live plot“-Selektor** für jede Methode: gestapelt pro Kanal, überlagert,
  FFT/Spektrum oder Einzelkanal.
- Bei „Simulate run“ streamt die Vorschau aus dem simulierten Gerät und sagt das auch.
- Die Live-Ansicht **referenziert ihren Puffer** und lässt Nicht-EEG-Spalten weg – zuvor
  wurden UNICORN-Beschleunigung/Batterie/Zähler auf einer µV-Achse geplottet.
  *(Commits `72bc2a2`, `3f882db`, `4cc1f9c`)*

---

## 7. Signalverarbeitung & Kanal-Identität

- **Einheitliche Kanal-Zuordnung:** Neues Modul `cortipy/shared/channels.py` ersetzt fünf
  abweichende Kopien. Die SSVEP-Variante verglich stringifizierte Dicts und traf daher nie –
  der PSD-Plot zeichnete immer Spalte 0, betitelte sich aber „Oz“. Plots benennen jetzt den
  **tatsächlich gezeichneten** Kanal und melden, wenn ein Ort fehlt (UNICORN hat kein Oz).
- **GND nicht mehr im EEG-Block:** `build_channels()` gibt jetzt (EEG-Kanäle, GND/Ref)
  getrennt zurück. Zuvor wurde GND vorangestellt, sodass 3 gewählte Elektroden **4 Kanäle**
  meldeten und Spalte 0 überall „GND“ hieß.
- **Referenzierung auf jedem Gerät** angewandt (nicht nur ActiCHamp/UNICORN): Simulierte und
  wiedergegebene Läufe blieben sonst unreferenziert und widersprachen einer Live-Aufnahme
  desselben Signals. Ein Referenzkanal außerhalb des Bereichs **warnt** jetzt, statt still
  Kanal 1 zu verwenden.
- **Simulierte Läufe zeigen echtes Signal:** `simulated_recording_data()` lieferte
  `np.zeros()` – „Simulate run“ erzeugte eine flache Linie, jedes Diagramm war leer. Jetzt
  wird echtes EEG erzeugt und die Stimulusfrequenz der Methode aufmoduliert.
- **SSVEP-Plot lesbar:** Fester Fallback auf Spalte 0 (= Referenz, identisch null) entfernt;
  feste `xlim=(0,500)`/`ylim` entfernt – bei fs=250 endet das Spektrum bei 125 Hz. Achsen
  folgen jetzt den Daten. *(Commits `4cc1f9c`, `8266b2e`, `3f882db`)*

**Pflicht-Stimulusfrequenzen:** StimFreq durfte auf 0 stehen, während der SSVEP-Evaluator
still 10 Hz einsetzte – ein Lauf konnte gegen eine Frequenz ausgewertet werden, die niemand
gewählt hat. `validate_params` blockiert das jetzt, das Formular warnt direkt am Feld.
*(Commit `3f882db`)*

---

## 8. Physikalische Einheiten (µV / V)

**Problem:** Jedes `RawArray` wurde direkt aus dem µV-Array gebaut, ein 50-µV-Signal also als
**50 V** gespeichert – ein Faktor-`1e6`-Fehler, der in jeden BIDS/EDF/Parquet-Export gelangte.

**Lösung:** Neues Modul `cortipy/shared/units.py`. Umgerechnet wird nur an der MNE-Grenze,
Stim-Kanäle werden nie skaliert, EDF deklariert seine Dimension, Nutzdaten bleiben in µV,
und `SignalUnit` hält die Konvention fest. Alt-Exporte (µV-Werte in Volt-Container) werden
erkannt und **nicht doppelt** skaliert. *(Commits `4cc1f9c`, `8266b2e`)*

---

## 9. Export: Dateinamen, BIDS/JSON-LD, Prüfsummen

- **Dateinamen-Überschreiben behoben:** Exporte landeten stets auf demselben Dateinamen mit
  `overwrite=True` – eine zweite Aufnahme in denselben Ordner zerstörte die erste. Läufe
  werden jetzt pro Namensstamm **nummeriert**, und der auf der Sitzungsseite eingegebene
  Dateiname wird (bereinigt) tatsächlich verwendet. *(Commit `72bc2a2`)*
- **SBIDS/JSON-LD-Metadaten vollständig:** Montage-Details (Position/Impedanz/Rubrik/Modell)
  und `ReferenceElectrodes` gingen beim Export verloren; GND/Ref sind jetzt AUXChannel-Knoten
  und blähen `NumberEEGChannels` nicht mehr auf. Gerät wird aus `schema:instrument` gelesen
  (nicht mehr fest „Offline“). **StimFreq/ASSR-Frequenzen** werden überhaupt serialisiert
  (zuvor verworfen, ein neu geladenes SSVEP wertete gegen den 10-Hz-Standard aus).
- **SHA-256** pro Datei-Knoten plus `verify_sbids_checksums()` und ein CLI – vorher wurde nur
  `fileSize` erfasst, was Korruption nicht erkennt.
- Doppeltes `sub-`-Präfix im Graph korrigiert; `contentUrl` nutzt auf Windows Vorwärts-
  Schrägstriche. *(Commits `4cc1f9c`, `8266b2e`)*

---

## 10. Navigation & Sitzungsverwaltung

- **Tabs „Workflow“ und „Preview“ ausgeblendet** (Code bleibt erhalten; eine einzige
  `HIDDEN_VIEWS`-Liste steuert das). Startseite ist jetzt „Session configuration“.
- **Laden erzwingt keine Offline-Wiedergabe mehr:** Das Laden einer Aufnahme deaktivierte
  „Connect device“ und machte ein erneutes Messen ohne Neustart unmöglich. Wiedergabe ist
  jetzt **opt-in**; das Abwählen stürzt die Seite nicht mehr ab (`bool()` auf einem
  2-D-Array). *(Commits `a0581b4`, `4cc1f9c`)*

---

## 11. Windows: echte Bluetooth-Gerätenamen

**Problem:** Windows benennt jeden Bluetooth-SPP-Port nach dem Treiber
(„Standardmäßige serielle über Bluetooth-Verbindung (COM12)“, Hersteller „Microsoft“) –
jedes gekoppelte Headset sah identisch aus, nichts kennzeichnete ein UNICORN.

**Lösung:** Neues Modul `serial_ports` liest den **echten Gerätenamen** (z. B.
`UN-2021.05.05`) vom Elternknoten im PnP-Baum (per PowerShell `Get-PnpDevice`), beschriftet
jeden Port, verwirft die nutzlosen „Microsoft“/„n/a“-Platzhalter und reiht wahrscheinliche
UNICORNs nach vorn. Die (langsame) Portliste wird über Reruns **zwischengespeichert**,
„Rescan“ verwirft den Cache. 27 Tests fixieren das Parsing anhand der echten deutschen
Ausgabe. Auf der Messmaschine zeigt `python -m cortipy.ui_streamlit.serial_ports`, was das
Dropdown anzeigen wird. *(Commit `cb8f7d2`)*