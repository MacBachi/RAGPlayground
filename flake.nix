# 1) Kopiere diese flake.nix in deinen Projektordner
# 2) Betritt die nix-shell mit 'nix develop'
# (automatisiert) 3) Erstelle das venv: 'python -m venv .venv'
# (automatisiert) 4) Aktiviere das venv 'source .venv/bin/activate'
# 5) installiere pakete mit pip 'pip install pandas jupyterlab requests'
# 6) Du verlässt die Umgebung mit dem Standard-Befehl zum Beenden einer Shell: exit

# ./flake.nix
{
  description = "Eine professionelle Python-Entwicklungsumgebung";

  # 1. EINGÄNGE: Definiert, woher der Code kommt (hier: nixpkgs).
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  # 2. AUSGÄNGE: Definiert, was diese Flake bereitstellt (hier: eine devShell).
  outputs =
    { self, nixpkgs }:
    let
      # Definiert das System. Passen Sie dies bei Bedarf an.
      # "aarch64-darwin" für Apple Silicon (M1/M2/M3)
      # "x86_64-darwin" für Intel-basierte Macs
      system = "aarch64-darwin";

      # Erstellt einen Verweis auf das nixpkgs-Set für das definierte System.
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      # Definiert die Entwicklungsumgebung (Development Shell).
      devShells.${system}.default = pkgs.mkShell {
        shell = pkgs.zsh;

        buildInputs = with pkgs; [
          black
          gcc
          git
          mypy
          poetry
          pkg-config
          pre-commit
          python313
          ruff
        ];

        shellHook = ''
          set -e
          echo "✅ Python-Entwicklungsumgebung (Python ${pkgs.python313.version}) ist aktiv."

          # Automatisches Erstellen und Aktivieren des venv
          if [ ! -d ".venv" ]; then
            echo "🐍 Erstelle virtuelles Environment im Ordner .venv..."
            python -m venv .venv
          fi
          source .venv/bin/activate

          # Poetry venv im Projektordner erzwingen (optional)
          poetry config virtualenvs.in-project true

          export MY_API_KEY="dein_schlüssel_hier"

          if [ -f ".pre-commit-config.yaml" ]; then
            pre-commit install
          fi
          
          if [ -f ".env" ]; then
            source .env
          fi
          
          if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
          fi
        '';
      };
    };
}
