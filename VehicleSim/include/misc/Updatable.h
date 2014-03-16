#ifndef VEHICLESIM_UPDATABLE
#define VEHICLESIM_UPDATABLE

namespace vlr
{
	class Updatable
	{
	public:
		Updatable()
			: _enabled(true)
		{

		}

		void setEnabled(bool val)
		{
			_enabled = val;
		}

		virtual void update() = 0;

	protected:
		bool _enabled;
	};
}

#endif /* VEHICLESIM_UPDATABLE */
